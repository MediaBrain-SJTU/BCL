import torch
import torch.nn as nn
import numpy as np
from models.resnet_prune_multibn import prune_resnet18_dual, prune_resnet50_dual, proj_head

class SDCLR(nn.Module):
    def __init__(self, num_class=100, network='resnet18'):
        super(SDCLR, self).__init__()
        self.backbone = self.get_backbone(network)(pretrained=False, imagenet=False, num_classes=num_class)
        self.backbone.in_planes = self.backbone.fc.weight.shape[1]
        self.backbone.fc = proj_head(self.backbone.in_planes, 128, twoLayerProj=True)

    @staticmethod
    def get_backbone(backbone_name):
        return {'resnet18': prune_resnet18_dual,
                'resnet50': prune_resnet50_dual}[backbone_name]

    def forward(self, x):
        
        x = self.backbone(x)

        return x


class Mask(object):
    def __init__(self, model, no_reset=False):
        super(Mask, self).__init__()
        self.model = model
        if not no_reset:
            self.reset()

    @property
    def sparsity(self):
        """Return the percent of weights that have been pruned as a decimal."""
        prunableTensors = []
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                prunableTensors.append(module.prune_mask.detach())

        unpruned = torch.sum(torch.tensor([torch.sum(v) for v in prunableTensors]))
        total = torch.sum(torch.tensor([torch.sum(torch.ones_like(v)) for v in prunableTensors]))
        return 1 - unpruned.float() / total.float()

    @property
    def density(self):
        return 1 - self.sparsity

    def magnitudePruning(self, magnitudePruneFraction, randomPruneFraction=0):
        weights = []
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                weights.append(module.weight.clone().cpu().detach().numpy())

        # only support one time pruning
        self.reset()
        prunableTensors = []
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                prunableTensors.append(module.prune_mask.detach())

        number_of_remaining_weights = torch.sum(torch.tensor([torch.sum(v) for v in prunableTensors])).cpu().numpy()
        number_of_weights_to_prune_magnitude = np.ceil(magnitudePruneFraction * number_of_remaining_weights).astype(int)
        number_of_weights_to_prune_random = np.ceil(randomPruneFraction * number_of_remaining_weights).astype(int)
        random_prune_prob = number_of_weights_to_prune_random / (number_of_remaining_weights - number_of_weights_to_prune_magnitude)

        # Create a vector of all the unpruned weights in the model.
        weight_vector = np.concatenate([v.flatten() for v in weights])
        threshold = np.sort(np.abs(weight_vector))[min(number_of_weights_to_prune_magnitude, len(weight_vector) - 1)]

        # apply the mask
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                module.prune_mask = (torch.abs(module.weight) >= threshold).float()
                # random weights been pruned
                module.prune_mask[torch.rand_like(module.prune_mask) < random_prune_prob] = 0

    def reset(self):
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                module.prune_mask = torch.ones_like(module.weight)

