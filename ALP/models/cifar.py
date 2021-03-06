


import torch.nn as nn
from .base import AdversarialDefensiveModule
from .layerops import Sequential




class CIFAR(AdversarialDefensiveModule):

    def __init__(self, dim_feature=256, num_classes=10):
        super(CIFAR, self).__init__()

        self.conv = Sequential(  # 3 x 32 x 32
            nn.Conv2d(3, 64, 3),  # 64 x 30 x 30
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),  # 64 x 28 x 28
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 x 14 x 14
            nn.Conv2d(64, 128, 3),  # 128 x 12 x 12
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3),  # 128 x 10 x 10
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 128 x 5 x 5
        )

        self.dense = Sequential(
            nn.Linear(128 * 5 * 5, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, dim_feature),
            nn.BatchNorm1d(dim_feature)
        )
        self.activation = nn.ReLU(inplace=True)
        self.fc = nn.Linear(dim_feature, num_classes)


    def forward(self, x):
        x = self.conv(x).flatten(start_dim=1)
        features = self.activation(self.dense(x))
        outs = self.fc(features)
        return outs

