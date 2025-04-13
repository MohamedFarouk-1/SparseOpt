import torch
import torch.nn as nn

class resnet18Optimized(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d()
        self.bn1 = BatchNorm2d()
        self.relu = ReLU()
        self.maxpool = MaxPool2d()
        self.layer1 = Module()
        self.layer2 = Module()
        self.layer3 = Module()
        self.layer4 = Module()
        self.avgpool = AdaptiveAvgPool2d()
        self.fc = Linear()
        
    def forward(self, x):
        return self.model(x)
