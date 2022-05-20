import torch
import torch.nn as nn
import os
import time
import numpy as np
import random
import torch.nn.functional as F
import re
import torchvision.transforms as transforms


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

class AverageMeter(object):
    """Computes and stores the average and current value"""
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

class logger(object):
    def __init__(self, path, log_name="log.txt", local_rank=0):
        self.path = path
        self.local_rank = local_rank
        self.log_name = log_name

    def info(self, msg):
        if self.local_rank == 0:
            print(msg)
            with open(os.path.join(self.path, self.log_name), 'a') as f:
                f.write(msg + "\n")

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def nt_xent(x, t=0.5, features2=None, average=True):

    if features2 is None:
        out = F.normalize(x, dim=-1)
        d = out.size()
        batch_size = d[0] // 2
        out = out.view(batch_size, 2, -1).contiguous()
        out_1 = out[:, 0]
        out_2 = out[:, 1]
    else:
        batch_size = x.shape[0]
        out_1 = F.normalize(x, dim=-1)
        out_2 = F.normalize(features2, dim=-1)

    # neg score
    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / t)

    mask = get_negative_mask(batch_size).cuda()
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    # pos score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / t)
    pos = torch.cat([pos, pos], dim=0)

    # estimator g()
    Ng = neg.sum(dim=-1)
    
    # contrastive loss
    loss = (- torch.log(pos / (pos + Ng)))
    loss_reshape = loss.view(2, batch_size).mean(0)

    if average:
        return loss.mean()
    else: 
        return loss_reshape


def cifar_aug():
    rnd_color_jitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    
    tfs_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.1, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        rnd_color_jitter,
        rnd_gray,
        transforms.ToTensor(),
    ])

    tfs_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    return tfs_train, tfs_test

def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi))

    return lr


