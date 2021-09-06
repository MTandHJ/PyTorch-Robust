






import torch.nn as nn
from .base import ADArch
from .layerops import Sequential



class MNIST(ADArch):
    def __init__(self, dim_features=200, num_classes=10, drop=0.5):
        super(MNIST, self).__init__()

        self.conv = Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )

        self.dense = Sequential(
            nn.Linear(64 * 4 * 4, 200),
            nn.ReLU(True),
            nn.Dropout(drop),
            nn.Linear(200, dim_features)
        )
        self.activation = nn.ReLU(True)
        self.fc = nn.Linear(dim_features, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.conv(x).flatten(start_dim=1)
        features = self.activation(self.dense(x))
        logits = self.fc(features)
        return logits
