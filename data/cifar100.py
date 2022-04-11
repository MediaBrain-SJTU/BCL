import torch
from torchvision.datasets import CIFAR100
from PIL import Image
import numpy as np

import torchvision.transforms as transforms
from utils import *
from data.randaug import *

strength = 1.0
resizeLower = 0.1
resizeHigher = 1.0

rnd_color_jitter = transforms.RandomApply([transforms.ColorJitter(0.4 * strength, 0.4 * strength,
                                                                  0.4 * strength, 0.1 * strength)], p=0.8)
rnd_gray = transforms.RandomGrayscale(p=0.2)

class CustomCIFAR100(CIFAR100):
    def __init__(self, sublist, **kwds):
        super().__init__(**kwds)
        self.txt = sublist

        if len(sublist) > 0:
            self.data = self.data[sublist]
            self.targets = np.array(self.targets)[sublist].tolist()

        self.idxsPerClass = [np.where(np.array(self.targets) == idx)[0] for idx in range(100)]
        self.idxsNumPerClass = [len(idxs) for idxs in self.idxsPerClass]

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        imgs = [self.transform(img), self.transform(img)]

        return torch.stack(imgs)

class CIFAR100_index(CIFAR100):
    def __init__(self, sublist, **kwds):
        super().__init__(**kwds)
        self.txt = sublist

        if len(sublist) > 0:
            self.data = self.data[sublist]
            self.targets = np.array(self.targets)[sublist].tolist()

        self.idxsPerClass = [np.where(np.array(self.targets) == idx)[0] for idx in range(100)]
        self.idxsNumPerClass = [len(idxs) for idxs in self.idxsPerClass]

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        imgs = [self.transform(img), self.transform(img)]

        labels = self.targets[idx]

        return torch.stack(imgs), labels, idx

class CIFAR100_index_bcl(CIFAR100):
    def __init__(self, sublist, **kwds):
        super().__init__(**kwds)
        self.txt = sublist

        if len(sublist) > 0:
            self.data = self.data[sublist]
            self.targets = np.array(self.targets)[sublist].tolist()

        self.idxsPerClass = [np.where(np.array(self.targets) == idx)[0] for idx in range(100)]
        self.idxsNumPerClass = [len(idxs) for idxs in self.idxsPerClass]

        self.momentum_weight=np.empty(len(sublist))
        self.momentum_weight[:]=0

    def update_momentum_weight(self, current):
        self.momentum_weight = current

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')

        tfs_train_re = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(resizeLower, resizeHigher), interpolation=3),
                transforms.RandomHorizontalFlip(),
                rnd_color_jitter,
                rnd_gray,
                RandAugment_prob(1, 30*self.momentum_weight[idx]*np.random.rand(1), 1.0*self.momentum_weight[idx]),
                transforms.ToTensor(),
            ])

        imgs = [tfs_train_re(img), tfs_train_re(img)]

        labels = self.targets[idx]

        return torch.stack(imgs), labels, idx, self.momentum_weight[idx]

