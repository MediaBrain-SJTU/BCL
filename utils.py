import torch
import torch.nn as nn
import os
import time
import numpy as np
import random
import torch.nn.functional as F
import re


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

def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi))

    return lr

def save_checkpoint(state, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)
    
def disjoint_summary(prefix, bestAcc, classWiseAcc, currentStatistics=None,
                      noReturnAvg=False, returnValue=False, group=3, noGroup=False):

    accList = []
    fullVarianceList = []
    GroupVarienceList = []
    majorAccList = []
    moderateAccList = []
    minorAccList = []

    sortIdx = np.argsort(currentStatistics)
    idxsMajor = sortIdx[len(currentStatistics) // 3 * 2:]
    idxsModerate = sortIdx[len(currentStatistics) // 3 * 1: len(currentStatistics) // 3 * 2]
    idxsMinor = sortIdx[: len(currentStatistics) // 3 * 1]

    classWiseAcc = np.array(classWiseAcc)
    bestAcc = np.mean(classWiseAcc)
    majorAcc = np.mean(classWiseAcc[idxsMajor])
    moderateAcc = np.mean(classWiseAcc[idxsModerate])
    minorAcc = np.mean(classWiseAcc[idxsMinor])

    accList.append(bestAcc)
    majorAccList.append(majorAcc)
    moderateAccList.append(moderateAcc)
    minorAccList.append(minorAcc)
    fullVarianceList.append(np.std(classWiseAcc / 100))
    GroupVarienceList.append(np.std(np.array([majorAcc, moderateAcc, minorAcc]) / 100))

    if group > 3:
        assert len(classWiseAcc) % group == 0
        group_idx_list = [sortIdx[len(currentStatistics) // group * cnt: len(currentStatistics) // group * (cnt + 1)] \
                            for cnt in range(0, group)]
        group_accs = [np.mean(classWiseAcc[group_idx_list[cnt]]) for cnt in range(0, group)]
        outputStr = "{}: group accs are".format(prefix)
        for acc in group_accs:
            outputStr += " {:.02f}".format(acc)
        print(outputStr)

    if returnValue:
        return accList, majorAccList, moderateAccList, minorAccList, fullVarianceList, GroupVarienceList
    else:
        if noReturnAvg:
            outputStr = "{}: accs are".format(prefix)
            for acc in accList:
                outputStr += " {:.02f}".format(acc)
            print(outputStr)
            if not noGroup:
                outputStr = "{}: majorAccs are".format(prefix)
                for acc in majorAccList:
                    outputStr += " {:.02f}".format(acc)
                print(outputStr)
                outputStr = "{}: moderateAccs are".format(prefix)
                for acc in moderateAccList:
                    outputStr += " {:.02f}".format(acc)
                print(outputStr)
                outputStr = "{}: minorAccs are".format(prefix)
                for acc in minorAccList:
                    outputStr += " {:.02f}".format(acc)
            print(outputStr)
        else:
            print("{}: acc is {:.02f}+-{:.02f}".format(prefix, np.mean(accList), np.std(accList)))
            if not noGroup:
                print("{}: vaiance is {:.04f}+-{:.04f}".format(prefix, np.mean(fullVarianceList), np.std(fullVarianceList)))
                print("{}: GroupBalancenessList is {:.04f}+-{:.04f}".format(prefix, np.mean(GroupVarienceList), np.std(GroupVarienceList)))
                print("{}: major acc is {:.02f}+-{:.02f}".format(prefix, np.mean(majorAccList), np.std(majorAccList)))
                print("{}: moderate acc is {:.02f}+-{:.02f}".format(prefix, np.mean(moderateAccList), np.std(moderateAccList)))
                print("{}: minor acc is {:.02f}+-{:.02f}".format(prefix, np.mean(minorAccList), np.std(minorAccList)))

