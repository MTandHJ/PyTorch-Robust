
import torch
import torch.nn as nn
from .base import AdversarialDefensiveModule


class Sequential(nn.Sequential, AdversarialDefensiveModule): ...
class ModuleList(nn.ModuleList, AdversarialDefensiveModule): ...

class TriggerBN1d(AdversarialDefensiveModule):

    def __init__(self, num_features: int, **kwargs):
        super(TriggerBN1d, self).__init__()
        self.bn_first = nn.BatchNorm1d(num_features, **kwargs)
        self.bn_second = nn.BatchNorm1d(num_features, **kwargs)

        nn.init.constant_(self.bn_first.weight, 1.)
        nn.init.constant_(self.bn_first.bias, 0.)
        nn.init.constant_(self.bn_second.weight, 1.)
        nn.init.constant_(self.bn_second.bias, 0.)

    def forward(self, x: torch.Tensor):
        # bn_first, the main batch normalization for defending
        if self.defending:
            return self.bn_first(x)
        else:
            return self.bn_second(x)


class TriggerBN2d(AdversarialDefensiveModule):
    
    def __init__(self, num_features: int, **kwargs):
        super(TriggerBN2d, self).__init__()
        self.bn_first = nn.BatchNorm2d(num_features, **kwargs)
        self.bn_second = nn.BatchNorm2d(num_features, **kwargs)

        nn.init.constant_(self.bn_first.weight, 1.)
        nn.init.constant_(self.bn_first.bias, 0.)
        nn.init.constant_(self.bn_second.weight, 1.)
        nn.init.constant_(self.bn_second.bias, 0.)

    def forward(self, x):
        # bn_first, the main batch normalization for defending
        if self.defending:
            return self.bn_first(x)
        else:
            return self.bn_second(x)
