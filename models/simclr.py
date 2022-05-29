import torch.nn as nn
from models.resnet import resnet18, resnet50

class proj_head(nn.Module):
    def __init__(self, ch, output_cnt=None):
        super(proj_head, self).__init__()
        self.in_features = ch

        if output_cnt is None:
            output_cnt = ch

        self.fc1 = nn.Linear(ch, ch)
        self.bn1 = nn.BatchNorm1d(ch)
      
        self.fc2 = nn.Linear(ch, output_cnt, bias=False)
        self.bn2 = nn.BatchNorm1d(output_cnt)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)

        return x

class SimCLR(nn.Module):
    def __init__(self, num_class=100, network='resnet18'):
        super(SimCLR, self).__init__()
        self.backbone = self.get_backbone(network)(pretrained=False, imagenet=False, num_classes=num_class)
        self.backbone.in_planes = self.backbone.fc.weight.shape[1]
        self.backbone.fc = nn.Identity()

        self.projector = proj_head(self.backbone.in_planes, 128)

    @staticmethod
    def get_backbone(backbone_name):
        return {'resnet18': resnet18,
                'resnet50': resnet50}[backbone_name]

    def forward(self, x):
        
        x = self.backbone(x)
        x = self.projector(x)

        return x

