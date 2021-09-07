
"""
Reference:
yaodongyu: https://github.com/yaodongyu/TRADES/blob/master/models/wideresnet_update.py
"""



import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import ADArch, AdversarialDefensiveModule
from .layerops import Sequential


__all__ = ["WideResNet", "wrn_28_10", "wrn_34_10", "wrn_34_20"]

class BasicBlock(AdversarialDefensiveModule):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(AdversarialDefensiveModule):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(ADArch):
    def __init__(self, depth=34, num_classes=10, scale=10, widen_factor=10, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.bias.data.zero_()

        self._generateOptMeans(
            self.fc,
            dim_feature=nChannels[3],
            num_classes=num_classes,
            scale=scale
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(self.bn1(x))
        features = self.avg_pool(x).flatten(start_dim=1)
        features = torch.flatten(x, 1).unsqueeze(dim=1)
        logits = -(features - self.fc.weight).pow(2).sum(dim=-1) # N x K
        return logits
        

def wrn_28_10(num_classes=10, **kwargs):
    return WideResNet(depth=28, widen_factor=10, num_classes=num_classes, **kwargs)

def wrn_34_10(num_classes=10, **kwargs):
    return WideResNet(depth=34, widen_factor=10, num_classes=num_classes, **kwargs)

def wrn_34_20(num_classes=10, **kwargs):
    return WideResNet(depth=34, widen_factor=20, num_classes=num_classes, **kwargs)




if __name__ == "__main__":

    model = wrn_28_10()
    x = torch.randn(10, 3, 32, 32)
    outs = model(x)
    print(outs.size())

