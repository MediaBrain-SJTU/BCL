import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from models.utils import NormalizeByChannelMeanStd
from pdb import set_trace

__all__ = ['ResNet', 'prune_resnet18_dual', 'prune_resnet34_dual', 'prune_resnet50_dual', 'prune_resnet101_dual',
           'prune_resnet152_dual']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

### custom layers ####
class prune_conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(prune_conv2d, self).__init__(*args, **kwargs)
        self.prune_mask = torch.ones(list(self.weight.shape))
        self.prune_flag = False

    def forward(self, input):
        if not self.prune_flag:
            weight = self.weight
        else:
            weight = self.weight * self.prune_mask

        if torch.__version__ < "1.8.0":
            return self._conv_forward(input, weight)
        else:
            return self._conv_forward(input, weight, self.bias)

    def set_prune_flag(self, flag):
        self.prune_flag = flag


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return prune_conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return prune_conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    
class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, bn_names=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = batch_norm_multiple(norm_layer, planes, bn_names=bn_names)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = batch_norm_multiple(norm_layer, planes, bn_names=bn_names)
        self.downsample = downsample
        self.stride = stride
        self.prune_flag = False

    def forward(self, x):
        identity = x

        out = x[0]
        bn_name = x[1]

        # debug
        # print("bn_name: {}".format(bn_name))

        out = self.conv1(out)
        out = self.bn1([out, bn_name])

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2([out, bn_name])

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity[0]
        out = self.relu(out)

        return [out, bn_name]

    def set_prune_flag(self, flag):
        self.prune_flag = flag
        for module in [self.conv1, self.conv2]:
            module.set_prune_flag(flag)
        if self.downsample is not None:
            self.downsample.set_prune_flag(flag)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, bn_names=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = batch_norm_multiple(norm_layer, width, bn_names=bn_names)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = batch_norm_multiple(norm_layer, width, bn_names=bn_names)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = batch_norm_multiple(norm_layer, planes * self.expansion, bn_names=bn_names)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.prune_flag = False

    def set_prune_flag(self, flag):
        self.prune_flag = flag
        for module in [self.conv1, self.conv2, self.conv3]:
            module.set_prune_flag(flag)
        if self.downsample is not None:
            self.downsample.set_prune_flag(flag)

    def forward(self, x):
        identity = x

        out = x[0]
        bn_name = x[1]

        out = self.conv1(out)
        out = self.bn1([out, bn_name])

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2([out, bn_name])

        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3([out, bn_name])

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity[0]
        out = self.relu(out)

        return [out, bn_name]


class Downsample_multiple(nn.Module):
    def __init__(self, inplanes, planes, expansion, stride, norm_layer, bn_names=None):
        super(Downsample_multiple, self).__init__()
        self.conv = conv1x1(inplanes, planes * expansion, stride)
        self.bn = batch_norm_multiple(norm_layer, planes * expansion, bn_names=bn_names)
        self.prune_flag = False

    def forward(self, x):
        out = x[0]
        bn_name = x[1]
        # debug
        # print("adv attack: {}".format(flag_adv))
        # print("out is {}".format(out))

        out = self.conv(out)
        out = self.bn([out, bn_name])

        return [out, bn_name]

    def set_prune_flag(self, flag):
        self.prune_flag = flag
        for module in [self.conv, ]:
            module.set_prune_flag(flag)


class batch_norm_multiple(nn.Module):
    def __init__(self, norm, inplanes, bn_names=None):
        super(batch_norm_multiple, self).__init__()

        # if no bn name input, by default use single bn
        self.bn_names = bn_names
        if self.bn_names is None:
            self.bn_list = norm(inplanes)
            return

        len_bn_names = len(bn_names)
        self.bn_list = nn.ModuleList([norm(inplanes) for _ in range(len_bn_names)])
        self.bn_names_dict = {bn_name: i for i, bn_name in enumerate(bn_names)}
        return

    def forward(self, x):
        out = x[0]
        name_bn = x[1]

        if name_bn is None:
            out = self.bn_list(out)
        else:
            bn_index = self.bn_names_dict[name_bn]
            out = self.bn_list[bn_index](out)

        return out


class proj_head(nn.Module):
    def __init__(self, ch, output_ch=None, bn_names=["nonprune", "prune"], twoLayerProj=False):
        super(proj_head, self).__init__()
        self.in_features = ch
        self.twoLayerProj = twoLayerProj

        if output_ch is None:
            output_ch = ch

        self.fc1 = nn.Linear(ch, ch)
        self.bn1 = batch_norm_multiple(nn.BatchNorm1d, ch, bn_names)

        if not twoLayerProj:
            self.fc2 = nn.Linear(ch, ch, bias=False)
            self.bn2 = batch_norm_multiple(nn.BatchNorm1d, ch, bn_names)

            self.fc3 = nn.Linear(ch, output_ch, bias=False)
            self.bn3 = batch_norm_multiple(nn.BatchNorm1d, output_ch, bn_names)
        else:
            self.fc2 = nn.Linear(ch, output_ch, bias=False)
            self.bn2 = batch_norm_multiple(nn.BatchNorm1d, output_ch, bn_names)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, bn_name):
        # debug
        # print("adv attack: {}".format(flag_adv))

        x = self.fc1(x)
        x = self.bn1([x, bn_name])

        x = self.relu(x)

        x = self.fc2(x)
        x = self.bn2([x, bn_name])

        if not self.twoLayerProj:
            x = self.relu(x)

            x = self.fc3(x)
            x = self.bn3([x, bn_name])

        return x


class proj_head_dual(nn.Module):
    def __init__(self, ch, output_ch=None, twoLayerProj=False):
        super(proj_head_dual, self).__init__()
        self.in_features = ch
        self.twoLayerProj = twoLayerProj
        bn_names = ["nonprune", "prune"]

        if output_ch is None:
            output_ch = ch

        self.fc1 = nn.Linear(ch, ch)
        self.bn1 = batch_norm_multiple(nn.BatchNorm1d, ch, bn_names)

        self.fc1_prune = nn.Linear(ch, ch)
        self.bn1_prune = batch_norm_multiple(nn.BatchNorm1d, ch, bn_names)

        if not twoLayerProj:
            self.fc2 = nn.Linear(ch, ch, bias=False)
            self.bn2 = batch_norm_multiple(nn.BatchNorm1d, ch, bn_names)

            self.fc2_prune = nn.Linear(ch, ch, bias=False)
            self.bn2_prune = batch_norm_multiple(nn.BatchNorm1d, ch, bn_names)

            self.fc3 = nn.Linear(ch, output_ch, bias=False)
            self.bn3 = batch_norm_multiple(nn.BatchNorm1d, output_ch, bn_names)

            self.fc3_prune = nn.Linear(ch, output_ch, bias=False)
            self.bn3_prune = batch_norm_multiple(nn.BatchNorm1d, output_ch, bn_names)
        else:
            self.fc2 = nn.Linear(ch, output_ch, bias=False)
            self.bn2 = batch_norm_multiple(nn.BatchNorm1d, output_ch, bn_names)

            self.fc2_prune = nn.Linear(ch, ch, bias=False)
            self.bn2_prune = batch_norm_multiple(nn.BatchNorm1d, ch, bn_names)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, bn_name):
        # debug
        # print("adv attack: {}".format(flag_adv))
        if bn_name == "nonprune":
            x = self.fc1(x)
            x = self.bn1([x, bn_name])

            x = self.relu(x)

            x = self.fc2(x)
            x = self.bn2([x, bn_name])

            if not self.twoLayerProj:
                x = self.relu(x)

                x = self.fc3(x)
                x = self.bn3([x, bn_name])
        else:
            assert bn_name == "prune"
            x = self.fc1_prune(x)
            x = self.bn1_prune([x, bn_name])

            x = self.relu(x)

            x = self.fc2_prune(x)
            x = self.bn2_prune([x, bn_name])

            if not self.twoLayerProj:
                x = self.relu(x)

                x = self.fc3_prune(x)
                x = self.bn3_prune([x, bn_name])

        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, imagenet=False, bn_names=["nonprune", "prune"]):
        """
        :param bn_names: list, the name of bn that would be employed
        """

        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.prune_flag = False

        self.normalize = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.inplanes = 64
        self.dilation = 1
        self.bn_names = bn_names

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        if not imagenet:
            self.conv1 = prune_conv2d(3, self.inplanes, 3, 1, 1, bias=False)
            self.bn1 = batch_norm_multiple(norm_layer, self.inplanes, bn_names=bn_names)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.Identity()
        else:
            self.conv1 = prune_conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = batch_norm_multiple(norm_layer, self.inplanes, bn_names=bn_names)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], bn_names=self.bn_names)
        self.layer2 = self._make_layer(block, 128, layers[1], bn_names=self.bn_names,
                                       stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], bn_names=self.bn_names,
                                       stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], bn_names=self.bn_names,
                                       stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, bn_names=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Downsample_multiple(self.inplanes, planes, block.expansion, stride, norm_layer, bn_names=bn_names)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, bn_names=bn_names))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, bn_names=bn_names))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        # debug
        # print("bn name: {}".format(bn_name))
        if self.prune_flag:
            bn_name = "prune"
        else:
            bn_name = "nonprune"

        # normalize
        x = self.normalize(x)

        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1([x, bn_name])

        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1([x, bn_name])
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x[0])
        x = torch.flatten(x, 1)

        if isinstance(self.fc, proj_head) or isinstance(self.fc, proj_head_dual):
            x = self.fc(x, bn_name)
        else:
            x = self.fc(x)

        return x

    def set_prune_flag(self, flag):
        self.prune_flag = flag
        for module in [self.conv1,]:
            module.set_prune_flag(flag)
        for stage in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for layer in stage:
                if isinstance(layer, BasicBlock) or isinstance(layer, Bottleneck) or isinstance(layer, prune_conv2d):
                    layer.set_prune_flag(flag)

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def prune_resnet10_dual(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet10', BasicBlock, [1, 1, 1, 1], pretrained, progress,
                   **kwargs)


def prune_resnet18_dual(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def prune_resnet34_dual(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def prune_resnet50_dual(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def prune_resnet101_dual(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def prune_resnet152_dual(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


