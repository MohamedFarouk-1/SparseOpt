import torch
import torch.nn as nn

class GCNModelOptimized(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv()
        self.conv2 = GCNConv()
        
    def forward(self, x):
        return self.model(x)
