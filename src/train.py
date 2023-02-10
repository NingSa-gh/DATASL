import logging
import os
import random
import shutil
import time
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import collections
import matplotlib.pyplot as plt
import pickle
#上包内有请求参数的代码，不能直接调用
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR
import tqdm
from utils import configuration
from numpy import linalg as LA
from scipy.stats import mode
import datasets
import models
from sklearn.decomposition import PCA
best_prec1 = -1
distance_count = []
sigmoid_num = 4.5
pcanum = 1
count_shot5=[]
max_sum =[]
num_save =[]
args = configuration.parser_args()
def main():
    global args, best_prec1
    # args = configuration.parser_args()

    ### initial logger
    # log = setup_logger(args.save_path + '/training.log')
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # create model
    log.info("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=args.num_classes, remove_linear=args.do_meta_train)

    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = get_optimizer(model)

    if args.pretrain:
        pretrain = args.pretrain + '/checkpoint.pth.tar'
        if os.path.isfile(pretrain):
            log.info("=> loading pretrained weight '{}'".format(pretrain))
            checkpoint = torch.load(pretrain)
            model_dict = model.state_dict()
            params = checkpoint['state_dict']
            params = {k: v for k, v in params.items() if k in model_dict}
            model_dict.update(params)
            model.load_state_dict(model_dict)
        else:
            log.info('[Attention]: Do not find pretrained model {}'.format(pretrain))

    # resume from an exist checkpoint
    if os.path.isfile(args.save_path + '/checkpoint.pth.tar') and args.resume == '':
        args.resume = args.save_path + '/checkpoint.pth.tar'

    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # scheduler.load_state_dict(checkpoint['scheduler'])
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
        else:
            log.info('[Attention]: Do not find checkpoint {}'.format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    if args.evaluate:
        do_extract_and_evaluate(model, log)
        return

    if args.do_meta_train:
        sample_info = [args.meta_train_iter, args.meta_train_way, args.meta_train_shot, args.meta_train_query]
        train_loader = get_dataloader('train', not args.disable_train_augment, sample=sample_info)
    else:
        train_loader = get_dataloader('train', not args.disable_train_augment, shuffle=True)

    sample_info = [args.meta_val_iter, args.meta_val_way, args.meta_val_shot, args.meta_val_query]
    val_loader = get_dataloader('val', False, sample=sample_info)

    scheduler = get_scheduler(len(train_loader), optimizer)
    tqdm_loop = warp_tqdm(list(range(args.start_epoch, args.epochs)))
    for epoch in tqdm_loop:
        scheduler.step(epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, scheduler, log)
        # evaluate on meta validation set
        is_best = False
        if (epoch + 1) % args.meta_val_interval == 0:
            prec1 = meta_val(val_loader, model)
            log.info('Meta Val {}: {}'.format(epoch, prec1))
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if not args.disable_tqdm:
                tqdm_loop.set_description('Best Acc {:.2f}'.format(best_prec1 * 100.))

        # remember best prec@1 and save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            # 'scheduler': scheduler.state_dict(),
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, folder=args.save_path)

    # do evaluate at the end
    do_extract_and_evaluate(model, log)


def get_metric(metric_type):
    METRICS = {
        'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        'euclidean': lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).sum(2),
        'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
        'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
    }
    return METRICS[metric_type]


def metric_prediction(gallery, query, train_label, metric_type):
    gallery = gallery.view(gallery.shape[0], -1)
    query = query.view(query.shape[0], -1)
    distance = get_metric(metric_type)(gallery, query)
    predict = torch.argmin(distance, dim=1)
    predict = torch.take(train_label, predict)

    return predict


def meta_val(test_loader, model, train_mean=None):
    top1 = AverageMeter()
    model.eval()

    with torch.no_grad():
        tqdm_test_loader = warp_tqdm(test_loader)
        for i, (inputs, target) in enumerate(tqdm_test_loader):
            target = target.cuda(0, non_blocking=True)
            output = model(inputs, True)[0].cuda(0)
            if train_mean is not None:
                output = output - train_mean
            train_out = output[:args.meta_val_way * args.meta_val_shot]
            train_label = target[:args.meta_val_way * args.meta_val_shot]
            test_out = output[args.meta_val_way * args.meta_val_shot:]
            test_label = target[args.meta_val_way * args.meta_val_shot:]
            train_out = train_out.reshape(args.meta_val_way, args.meta_val_shot, -1).mean(1)
            train_label = train_label[::args.meta_val_shot]
            prediction = metric_prediction(train_out, test_out, train_label, args.meta_val_metric)
            acc = (prediction == test_label).float().mean()
            top1.update(acc.item())
            if not args.disable_tqdm:
                tqdm_test_loader.set_description('Acc {:.2f}'.format(top1.avg * 100))
    return top1.avg


def train(train_loader, model, criterion, optimizer, epoch, scheduler, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    tqdm_train_loader = warp_tqdm(train_loader)
    for i, (input, target) in enumerate(tqdm_train_loader):
        if args.scheduler == 'cosine':
            scheduler.step(epoch * len(train_loader) + i)
        # measure data loading time
        data_time.update(time.time() - end)

        if args.do_meta_train:
            target = torch.arange(args.meta_train_way)[:, None].repeat(1, args.meta_train_query).reshape(-1).long()
        target = target.cuda(non_blocking=True)

        # compute output
        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
            output = model(input)
            if args.do_meta_train:
                output = output.cuda(0)
                shot_proto = output[:args.meta_train_shot * args.meta_train_way]
                query_proto = output[args.meta_train_shot * args.meta_train_way:]
                shot_proto = shot_proto.reshape(args.meta_train_way, args.meta_train_shot, -1).mean(1)
                output = -get_metric(args.meta_train_metric)(shot_proto, query_proto)
            loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        if not args.disable_tqdm:
            tqdm_train_loader.set_description('Acc {:.2f}'.format(top1.avg))

        # measure elapsed time
        # measure elapsed tim
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log.info('Epoch: [{0}][{1}/{2}]\t'
                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                     'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', folder='result/default'):
    torch.save(state, folder + '/' + filename)
    if is_best:
        shutil.copyfile(folder + '/' + filename, folder + '/model_best.pth.tar')


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def setup_logger(filepath):
    file_formatter = logging.Formatter(
        "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)-8s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger('example')
    # handler = logging.StreamHandler()
    # handler.setFormatter(file_formatter)
    # logger.addHandler(handler)

    file_handle_name = "file"
    if file_handle_name in [h.name for h in logger.handlers]:
        return
    if os.path.dirname(filepath) is not None:
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
    file_handle = logging.FileHandler(filename=filepath, mode="a")
    file_handle.set_name(file_handle_name)
    file_handle.setFormatter(file_formatter)
    logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger


def get_scheduler(batches, optimiter):
    """
    cosine will change learning rate every iteration, others change learning rate every epoch
    :param batches: the number of iterations in each epochs
    :return: scheduler
    """
    SCHEDULER = {'step': StepLR(optimiter, args.lr_stepsize, args.lr_gamma),
                 'multi_step': MultiStepLR(optimiter, milestones=[int(.5 * args.epochs), int(.75 * args.epochs)],
                                           gamma=args.lr_gamma),
                 'cosine': CosineAnnealingLR(optimiter, batches * args.epochs, eta_min=1e-9)}
    return SCHEDULER[args.scheduler]


def get_optimizer(module):
    OPTIMIZER = {'SGD': torch.optim.SGD(module.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
                                        nesterov=args.nesterov),
                 'Adam': torch.optim.Adam(module.parameters(), lr=args.lr)}
    return OPTIMIZER[args.optimizer]


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def extract_feature(train_loader, val_loader, model, tag='last'):
    # return out mean, fcout mean, out feature, fcout features
    save_dir = '{}/{}/{}'.format(args.save_path, tag, args.enlarge)
    if os.path.isfile(save_dir + '/output.plk'):
        data = load_pickle(save_dir + '/output.plk')
        return data
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    model.eval()
    with torch.no_grad():
        # get training mean
        out_mean, fc_out_mean = [], []
        for i, (inputs, _) in enumerate(warp_tqdm(train_loader)):
            outputs, fc_outputs = model(inputs, True)
            out_mean.append(outputs.cpu().data.numpy())
            if fc_outputs is not None:
                fc_out_mean.append(fc_outputs.cpu().data.numpy())
        out_mean = np.concatenate(out_mean, axis=0).mean(0)
        if len(fc_out_mean) > 0:
            fc_out_mean = np.concatenate(fc_out_mean, axis=0).mean(0)
        else:
            fc_out_mean = -1

        output_dict = collections.defaultdict(list)
        fc_output_dict = collections.defaultdict(list)
        for i, (inputs, labels) in enumerate(warp_tqdm(val_loader)):
            # compute output
            outputs, fc_outputs = model(inputs, True)
            outputs = outputs.cpu().data.numpy()
            if fc_outputs is not None:
                fc_outputs = fc_outputs.cpu().data.numpy()
            else:
                fc_outputs = [None] * outputs.shape[0]
            for out, fc_out, label in zip(outputs, fc_outputs, labels):
                output_dict[label.item()].append(out)
                fc_output_dict[label.item()].append(fc_out)
        all_info = [out_mean, fc_out_mean, output_dict, fc_output_dict]
        save_pickle(save_dir + '/output.plk', all_info)
        return all_info


def get_dataloader(split, aug=False, shuffle=True, out_name=False, sample=None):
    # sample: iter, way, shot, query
    if aug:
        transform = datasets.with_augment(84, disable_random_resize=args.disable_random_resize)
    else:
        transform = datasets.without_augment(84, enlarge=args.enlarge)
    sets = datasets.DatasetFolder(args.data, args.split_dir, split, transform, out_name=out_name)
    # sets = datasets.myDataset(args.data, split, transform, out_name=out_name)
    if sample is not None:
        sampler = datasets.CategoriesSampler(sets.labels, *sample)
        loader = torch.utils.data.DataLoader(sets, batch_sampler=sampler,
                                             num_workers=args.workers, pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(sets, batch_size=args.batch_size, shuffle=shuffle,
                                             num_workers=args.workers, pin_memory=True)
    return loader


def warp_tqdm(data_loader):
    if args.disable_tqdm:
        tqdm_loader = data_loader
    else:
        tqdm_loader = tqdm.tqdm(data_loader, total=len(data_loader))
    return tqdm_loader


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def load_checkpoint(model, type='best'):
    if type == 'best':
        checkpoint = torch.load('{}/model_best.pth.tar'.format(args.save_path))
    elif type == 'last':
        checkpoint = torch.load('{}/checkpoint.pth.tar'.format(args.save_path))
    else:
        assert False, 'type should be in [best, or last], but got {}'.format(type)
    model.load_state_dict(checkpoint['state_dict'])


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def meta_evaluate(data, train_mean, shot, shift, t=0.05, do=False):
    un_list = []
    l2n_list = []
    cl2n_list = []
    for _ in warp_tqdm(range(args.meta_test_iter)):
        train_data, test_data, train_label, test_label = sample_case(data, shot)
        acc = metric_class_type(train_data, test_data, train_label, test_label, shot, do, shift, t, train_mean=train_mean,
                                norm_type='CL2N')
        cl2n_list.append(acc)
        acc = metric_class_type(train_data, test_data, train_label, test_label, shot, do, shift, t, train_mean=train_mean,
                                norm_type='L2N')
        l2n_list.append(acc)
        acc = metric_class_type(train_data, test_data, train_label, test_label, shot, do, shift, t, train_mean=train_mean,
                                norm_type='UN')
        un_list.append(acc)
    un_mean, un_conf = compute_confidence_interval(un_list)
    l2n_mean, l2n_conf = compute_confidence_interval(l2n_list)
    cl2n_mean, cl2n_conf = compute_confidence_interval(cl2n_list)
    return un_mean, un_conf, l2n_mean, l2n_conf, cl2n_mean, cl2n_conf


def cal_sim(x, y, idx):
    sim = 0
    for i in idx[x, :]:
        if i in idx[y, :]:
            sim += 1
    return sim


def get_logits(sim, ids, graph, pos, original):
    logit = []
    #1.查询点不跟任何支持点存在互近邻的可能；
    if len(ids) == 0:
        index = 0
        a, idx1 = torch.sort(original, dim=1, descending=True)
        for i in idx1[pos, :]:
            if i >= 75:
                index = i
                break
        for i in range(75, original.shape[0]):
            if i == index:
                logit.append(1.0)
            else:
                logit.append(0.1)
        return logit

    maxx = np.max(sim)
    count = np.sum(sim == maxx)
    # 2.查询点有许多相似度不同的互近邻支持点；
    # if count == 1:
    for i in range(75, original.shape[0]):
        mid = 0.1
        if i in ids:
            mid = float(sim[ids.index(i)])
        # if i == index:
        #     mid += 1
        logit.append(mid)
    return logit

    # 3.查询点跟许多支持点互近邻，但存在互近邻点相似度相等的情况;
    indexs = []
    maxid = 0
    for i in range(len(sim)):
        if sim[i] == maxx:
            indexs.append(ids[i])
    for i in graph[pos, :]:
        if i in indexs:
            maxid = i
            break
    for i in range(75, original.shape[0]):
        mid = 0.1
        if i in ids:
            mid = float(sim[ids.index(i)])
        if i == maxid:
            mid += 1
        logit.append(mid)
    return logit

def CenterRectification(cent, query):
    subtract = cent[:, None, :] - query
    distance = LA.norm(subtract, 2, axis=-1)  # 5 * 100

    prob = []
    for p in range(75):
        total = 0.
        for ob in range(5):
            obj = distance[ob, p]
            total += np.exp(-np.square(obj))
        pr = []
        for ob in range(5):
            pr.append(np.exp(-np.square(distance[ob, p])) / total)
        prob.append(pr)
    prob = np.array(prob)
    K = [[], [], [], [], []]
    Kp = [[], [], [], [], []]
    P = [[], [], [], [], []]
    midgallery = []
    D = [[], [], []]
    Kp_PD = [[], [], [], [], []]
    K_PD = [[], [], [], [], []]
    P_PD = [[], [], [], [], []]
    Kp_nor = [[], [], [], [], []]
    K_nor = [[], [], [], [], []]
    Pp_nor = [[], [], [], [], []]
    Score_nor = [[], [], [], [], []]
    PD_avg = 0
    for k in range(75):
        rank = np.argsort(prob[k, :])[-2:]
        pr = prob[k, rank[1]] - prob[k, rank[0]]
        PD_avg += pr
        po = rank[1]
        K[po].append(k)
        Kp[po].append(pr)
        P[po].append(prob[k][po])
        D[0].append(k)
        D[1].append(prob[k][po])
        D[2].append(pr)

    for i in range(5):
        if len(Kp[i]) < 2:
            Kp_nor[i] = Kp[i]
            Pp_nor[i] = P[i]
            continue
        if (np.array(Kp[i]).max() - np.array(Kp[i]).min()) == 0:
            Kp_nor[i] = Kp[i]
            Pp_nor[i] = P[i]
            continue
        Kp_nor[i] = [((X - np.array(Kp[i]).min()) / (np.array(Kp[i]).max() - np.array(Kp[i]).min())) / 2 for X
                     in Kp[i]]
        if (np.array(P[i]).min() - np.array(P[i]).max()) == 0:
            Pp_nor[i] = [0.5 for X in range(len(P[i]))]
        else:
            Pp_nor[i] = [((X - np.array(P[i]).min()) / (np.array(P[i]).max() - np.array(P[i]).min())) / 2 for X in
                         P[i]]
        Score_nor[i] = [Kp_nor[i][j] + Pp_nor[i][j] for j in range(len(Kp[i]))]

    mink = 100
    Kpos = []
    Ppos = []
    for i in range(5):
        if len(K[i]) < mink:
            mink = len(K[i])
        mid = np.array(Kp[i])
        Kpos.append(mid.argsort())
        mid = np.array(P[i])
        Ppos.append(mid.argsort())
    if mink == 0:
        return cent

    Fgallery = []
    Pesoudo = []
    PD_mink = [[], [], [], [], []]
    P_mink = [[], [], [], [], []]
    for i in range(5):
        middle = []
        for j in range(1, mink + 1):
            mid = Kpos[i][-j]
            PD_mink[i].append(Kp[i][mid])

            mid = K[i][mid]
            mid = query[mid, :]
            middle.append(mid)
        Pesoudo.append(middle)
        # for k in range(shot):
        #     middle.append(gallery[i*shot + k])
        # Fgallery.append(middle)
    Wkp = [[], [], [], [], []]
    Pesoudo_W = [[], [], [], [], []]
    for i in range(5):
        for j in range(mink):
            Wkp[i].append(Kp_nor[i][j] + Pp_nor[i][j])
            Pesoudo_W[i].append(Wkp[i][j] * Pesoudo[i][j])
    Pesoudo = Pesoudo_W

    for i in range(5):
        mid = Pesoudo[i]
        mid = np.array(mid)
        midgallery.append(mid.mean(0))
    cent_esoudo = np.array(midgallery)
    cent = cent / 2 + cent_esoudo / 2
    return cent


def count_lesszero(array):
    num = 0
    for x in array:
        if x < 0:
            num += 1
    return num


def metric_class_type(gallery, query, train_label, test_label, shot, do, shift, threshold=0.06, train_mean=None, norm_type='CL2N'):

    if norm_type == 'CL2N':
        gallery = gallery - train_mean
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
    elif norm_type == 'UN':
        query = query - train_mean
        query = query / LA.norm(query, 2, 1)[:, None]
    elif norm_type == 'L2N':
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query / LA.norm(query, 2, 1)[:, None]

    # gallery = gallery.reshape(args.meta_val_way, shot, gallery.shape[-1])
    # cent = gallery.reshape(args.meta_val_way, shot, gallery.shape[-1]).mean(1)

    train_label = train_label[::shot]

    datas = []
    datas.extend(gallery)
    datas.extend(query)
    query_ori = query
    gallery_ori = gallery


    mean_q = query_ori.mean(0)
    mean_s = gallery_ori.mean(0)
    dert = mean_s - mean_q
    query_ori = query + dert
    datas = []
    # gallery_ori = cent
    datas.extend(gallery_ori)
    datas.extend(query_ori)
    gallery_ori = gallery




    start = 1
    end = 30
    num = 15
    pos = shot * 5
    if shot == 1:
        start = 1
        num = 15


    if do:
        distances = []
        variances = []
        variances_sum = []
        variances_ratio = []
        variances_ratio_sum = []
        max_distances_cen = []
        max_distances_p2p = []
        new_idistance = []
        for k in range(start, end):
            model = PCA(n_components=k)
            model = model.fit(datas)
            variances.append(model.explained_variance_)
            variances_ratio_sum.append(np.sum(variances[len(variances) - 1]))
            transdatas = model.transform(datas)
            pos = len(gallery_ori)
            gallery = transdatas[:pos]
            query = transdatas[pos:]
            gallery = gallery.reshape(args.meta_val_way, shot, gallery.shape[-1]).mean(1)
            for i in range(4):
                gallery = CenterRectification(gallery, query)
            # 对质心和查询
            subtract = gallery[:, None, :] - query
            distance = LA.norm(subtract, 2, axis=-1)
            max_distance = distance.max(axis=0)
            min_distance = distance.min(axis=0)
            dif = max_distance / min_distance
            isubtract = transdatas[:, None, :] - transdatas
            idistance = LA.norm(isubtract, 2, axis=-1)
            for i in range(len(idistance[0])):
                new_idistance.append(np.delete(idistance[i], i))
            idistance = np.array(new_idistance)
            avg_distance_75 = np.array([np.sum(X)/5 for X in distance.T])
            min_cen = np.min(avg_distance_75)
            max_cen = np.max(avg_distance_75)
            # max_distances_cen.append(max_cen / min_cen)

            avg_distance_5 = np.array([X for X in distance.T])
            min_cen = np.array([np.min(X) for X in avg_distance_5])
            max_cen = np.array([np.max(X) for X in avg_distance_5])
            avg_cen = np.array([max_cen[X] / min_cen[X] for X in range(len(min_cen))])
            max_distances_cen.append(avg_cen.mean(0))

            if np.isnan(max_distances_cen[len(max_distances_cen)-1]):
                max_distances_cen[len(max_distances_cen) - 1] = max_distances_cen[len(max_distances_cen)-2] + (max_distances_cen[len(max_distances_cen)-3] - max_distances_cen[len(max_distances_cen)-2])
            min_p2p = np.min(idistance)
            max_p2p = np.max(idistance)
            max_distances_p2p.append(max_p2p / min_p2p)
            distances.append(dif)
        max_distances_cen = np.array(max_distances_cen)

        # global sigmoid_num
        # idistance = np.array([1.0 / (1. + np.exp(-X * sigmoid_num)) for X in max_distances_cen])
        # ivariance = np.array([1.0 / (1. + np.exp(-X/100 * sigmoid_num)) for X in variances_ratio_sum])
        # num = int(np.argmax(np.array([idistance[c] + ivariance[c] for c in range(len(ivariance))])))


        diff_var = np.diff(variances[len(variances)-1]) / 1
        diff_dis = np.diff(max_distances_cen)

        sec_dis = np.diff(diff_dis) / 1
        sec_var = np.diff(diff_var)
        var_num = 0
        dis_num = 0
        num1 = -1
        num2 = -1
        num3 = -1
        value_range = 5
        for j in range(end - start - 2 - value_range):
            if sec_var[j] < 0:
                test_array = [sec_var[j + X] for X in range(value_range)]
                if count_lesszero(test_array) >= value_range - 2 and j + value_range <= (end - start - 2 - value_range):
                    var_num = j+4
                    for k in range(var_num, end - start - 2 - value_range):
                        if sec_dis[k] < 0:
                            test_array = [sec_dis[k + X] for X in range(value_range)]
                            if count_lesszero(test_array) >= value_range - 2:
                                num1 = k+4
                                break
                    if num1 != -1:
                        break
        for j in range(end - start - 2 - value_range):
            if sec_dis[j] < 0:
                test_array = [sec_dis[j + X] for X in range(value_range)]
                if count_lesszero(test_array) >= value_range - 2 and j + value_range <= (end - start - 2 - value_range):
                    var_num = j+4
                    for k in range(var_num, end - start - 2 - value_range):
                        if sec_var[k] < 0:
                            test_array = [sec_var[k + X] for X in range(value_range)]
                            if count_lesszero(test_array) >= value_range - 2:
                                num2 = k+4
                                break
                    if num2 != -1:
                        break
        for j in range(end - start - 2 - value_range):
            if sec_var[j] < 0:
                test_array = [sec_var[j + X] for X in range(value_range)]
                if count_lesszero(test_array) >= value_range - 2 and j + value_range <= (end - start - 2 - value_range):
                    var_num = j+4
        for k in range(end - start - 2 - value_range):
            if sec_dis[k] < 0:
                test_array = [sec_dis[k + X] for X in range(value_range)]
                if count_lesszero(test_array) >= value_range - 2:
                    var_num_1 = k+4
                    # if var_num > var_num_1:
                    #     num3 = var_num
                    # else:
                    #     num3 = var_num_1
                    num3 = int((var_num_1 + var_num)/2)
                    break
                if num3 != -1:
                    break

        if num1 != -1 and num2 != -1:
            # num = int((num1 + num2)/2)
            num = num3

    global pcanum
    count_model = PCA(n_components=num)
    matter_transdatas = count_model.fit_transform(datas)

    # 添加L2归一化
    new_matter_transdatas = matter_transdatas / LA.norm(matter_transdatas, 2, 1)[:, None]
    gallery = new_matter_transdatas[:pos]
    query = new_matter_transdatas[pos:]
    cent = gallery.reshape(args.meta_val_way, shot, gallery.shape[-1]).mean(1)


    if do:
        for z in range(4):
            #simpleshot way
            subtract = cent[:, None, :] - query
            distance = LA.norm(subtract, 2, axis=-1) # 5 * 100
            prob = []
            for p in range(75):
                total = 0.
                for ob in range(5):
                    obj = distance[ob, p]
                    total += np.exp(-np.square(obj))
                pr = []
                for ob in range(5):
                    pr.append(np.exp(-np.square(distance[ob, p])) / total)
                prob.append(pr)
            prob = np.array(prob)

            '''
            实验一 根据PD值的大小添加mink，分别单一类内的P和PD归一化，之后使用归一化之后的P+PD作为权重求质心
            '''
            K = [[], [], [], [], []]
            Kp = [[], [], [], [], []]
            P = [[], [], [], [], []]
            midgallery = []
            D = [[], [], []]
            Kp_nor = [[], [], [], [], []]
            Pp_nor = [[], [], [], [], []]
            Score_nor = [[], [], [], [], []]
            PD_avg = 0
            for k in range(75):
                rank = np.argsort(prob[k, :])[-2:]
                pr = prob[k, rank[1]] - prob[k, rank[0]]
                PD_avg += pr
                po = rank[1]
                K[po].append(k)
                Kp[po].append(pr)
                P[po].append(prob[k][po])
                D[0].append(k)
                D[1].append(prob[k][po])
                D[2].append(pr)

            for i in range(5):
                if len(Kp[i]) < 2:
                    Kp_nor[i] = Kp[i]
                    Pp_nor[i] = P[i]
                    continue
                if (np.array(Kp[i]).max() - np.array(Kp[i]).min()) == 0:
                    Kp_nor[i] = Kp[i]
                    Pp_nor[i] = P[i]
                    continue

                Kp_nor[i] = [((X - np.array(Kp[i]).min()) / (np.array(Kp[i]).max() - np.array(Kp[i]).min())) / 2 for X
                             in Kp[i]]
                if (np.array(Kp[i]).min() - np.array(Kp[i]).max()) == 0:
                    Pp_nor[i] = [0.5 for X in range(len(P[i]))]
                else:
                    Pp_nor[i] = [((X - np.array(P[i]).min()) / (np.array(P[i]).max() - np.array(P[i]).min())) / 2 for X in
                             P[i]]
                Score_nor[i] = [Kp_nor[i][j] + Pp_nor[i][j] for j in range(len(Kp[i]))]

            mink = 100
            Kpos = []
            Ppos = []
            for i in range(5):
                if len(K[i]) < mink:
                    mink = len(K[i])
                mid = np.array(Kp[i])
                Kpos.append(mid.argsort())
                mid = np.array(P[i])
                Ppos.append(mid.argsort())
            if mink == 0:
                break

            Pesoudo = []
            PD_mink = [[], [], [], [], []]
            for i in range(5):
                middle = []
                for j in range(1, mink + 1):
                    mid = Kpos[i][-j]
                    PD_mink[i].append(Kp[i][mid])
                    mid = K[i][mid]
                    mid = query[mid, :]
                    middle.append(mid)
                Pesoudo.append(middle)
            Wkp = [[], [], [], [], []]
            Pesoudo_W = [[], [], [], [], []]
            for i in range(5):
                for j in range(mink):
                    Wkp[i].append(Kp_nor[i][j] + Pp_nor[i][j])
                    Pesoudo_W[i].append(Wkp[i][j] * Pesoudo[i][j])
            Pesoudo = Pesoudo_W

            for i in range(5):
                mid = Pesoudo[i]
                mid = np.array(mid)
                midgallery.append(mid.mean(0))
            cent_esoudo = np.array(midgallery)
            cent = cent / 2 + cent_esoudo / 2



    query = matter_transdatas

    # 添加L2归一化
    new_matter_transdatas = []
    new_matter_transdatas.extend(cent)
    new_matter_transdatas.extend(query)
    new_matter_transdatas = np.array(new_matter_transdatas)
    new_matter_transdatas = new_matter_transdatas / LA.norm(new_matter_transdatas, 2, 1)[:, None]
    cent = new_matter_transdatas[:5]
    query = new_matter_transdatas[5:]

    subtract = cent[:, None, :] - query[5 * shot:, :]
    distance = LA.norm(subtract, 2, axis=-1)

    idx1 = np.argpartition(distance, args.num_NN, axis=0)[:args.num_NN] #找出最近的欧式质心序号
    # idx = np.argmax(similarity, axis=1) # cosine质心匹配

    nearest_samples = np.take(train_label, idx1)  # 提取质心序号相应的标签
    out = nearest_samples
    out = out.astype(int)
    test_label = np.array(test_label)
    acc = (out == test_label).mean()
    return acc


def sample_case(ld_dict, shot):
    sample_class = random.sample(list(ld_dict.keys()), args.meta_val_way)
    train_input = []
    test_input = []
    test_label = []
    train_label = []
    for each_class in sample_class:
        samples = random.sample(ld_dict[each_class], shot + args.meta_val_query)
        train_label += [each_class] * len(samples[:shot])
        test_label += [each_class] * len(samples[shot:])
        train_input += samples[:shot]
        test_input += samples[shot:]
    train_input = np.array(train_input).astype(np.float32)
    test_input = np.array(test_input).astype(np.float32)
    return train_input, test_input, train_label, test_label


def do_extract_and_evaluate(model, log):
    train_loader = get_dataloader('train', aug=False, shuffle=False, out_name=False)
    val_loader = get_dataloader('test', aug=False, shuffle=False, out_name=False)
    t = 0.
    shift = 1
    # for t in np.arange(0., 0.01, 0.001):
    # for shift in range(1, 4):
    # load_checkpoint(model, 'last')
    # print('---------------------undo----------------- ' + str(t) + ' - ' + str(shift))
    # out_mean, fc_out_mean, out_dict, fc_out_dict = extract_feature(train_loader, val_loader, model, 'last')
    # accuracy_info_shot1 = meta_evaluate(out_dict, out_mean, 1, shift, t)
    # accuracy_info_shot5 = meta_evaluate(out_dict, out_mean, 5, shift, t)
    # print(
    #     'Meta Test: LAST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format(
    #         'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))
    # log.info(
    #     'Meta Test: LAST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format(
    #         'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))
    #
    #
    # load_checkpoint(model, 'best')
    # out_mean, fc_out_mean, out_dict, fc_out_dict = extract_feature(train_loader, val_loader, model, 'best')
    # accuracy_info_shot1 = meta_evaluate(out_dict, out_mean, 1, shift, t)
    # accuracy_info_shot5 = meta_evaluate(out_dict, out_mean, 5, shift, t)
    # print(
    #     'Meta Test: BEST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format(
    #         'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))
    # log.info(
    #     'Meta Test: BEST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format(
    #         'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))

    print('---------------------do-----------------')
    # load_checkpoint(model, 'last')
    # out_mean, fc_out_mean, out_dict, fc_out_dict = extract_feature(train_loader, val_loader, model, 'last')
    # accuracy_info_shot1 = meta_evaluate(out_dict, out_mean, 1, shift, t, True)
    # accuracy_info_shot5 = meta_evaluate(out_dict, out_mean, 5, shift, t, True)
    # print(
    #     'Meta Test: LAST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})  shift: {:.1f}'.format(
    #         'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5, shift))
    # log.info(
    #     'Meta Test: LAST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})  shift: {:.1f}'.format(
    #         'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5, shift))

    load_checkpoint(model, 'best')
    out_mean, fc_out_mean, out_dict, fc_out_dict = extract_feature(train_loader, val_loader, model, 'best')
    accuracy_info_shot1 = meta_evaluate(out_dict, out_mean, 1, shift, t, True)
    accuracy_info_shot5 = meta_evaluate(out_dict, out_mean, 5, shift, t, True)
    print(
        'Meta Test: BEST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})  sigmoid_num: {:.2f}'.format(
            'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5, sigmoid_num))
    log.info(
        'Meta Test: BEST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})  sigmoid_num: {:.2f}'.format(
            'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5, sigmoid_num))


if __name__ == '__main__':
    log = setup_logger(args.save_path + '/training.log')
    # main()
    while 1:
        main()
        # pcanum += 1
        sigmoid_num += 0.5
        if sigmoid_num > 5:
            break
