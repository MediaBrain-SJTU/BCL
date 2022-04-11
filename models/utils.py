import torch
import torch.nn as nn
from torch.nn.functional import feature_alpha_dropout
import torch.nn.functional as F


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


class proj_head(nn.Module):
    def __init__(self, ch, output_cnt=None, finetuneMode=False):
        super(proj_head, self).__init__()
        self.in_features = ch
        self.finetuneMode = finetuneMode

        if output_cnt is None:
            output_cnt = ch

        self.fc1 = nn.Linear(ch, ch)
        self.bn1 = nn.BatchNorm1d(ch)

        if not self.finetuneMode:
            self.fc2 = nn.Linear(ch, output_cnt, bias=False)
            self.bn2 = nn.BatchNorm1d(output_cnt)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if not self.finetuneMode:
            x = self.fc2(x)
            x = self.bn2(x)

        return x


class proj_regression_head(nn.Module):
    def __init__(self, ch, output_cnt=None, finetuneMode=False):
        super(proj_regression_head, self).__init__()
        self.in_features = ch
        self.finetuneMode = finetuneMode

        if output_cnt is None:
            output_cnt = ch

        self.fc1 = nn.Linear(ch, ch)
        self.bn1 = nn.BatchNorm1d(ch)

        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(ch, output_cnt, bias=False)
        self.bn2 = nn.BatchNorm1d(output_cnt)  

    def forward(self, x):

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)

        return x



