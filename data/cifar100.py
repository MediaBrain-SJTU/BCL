import torch
from torchvision.datasets import CIFAR100
from PIL import Image
import numpy as np

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

        return torch.stack(imgs), idx


