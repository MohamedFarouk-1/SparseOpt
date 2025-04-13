import torch
import torch.nn as nn

class SimpleSparseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.8)
        self.linear2 = nn.Linear(512, 512)

    def forward(self, x):
        x = self.linear1(x)
        if torch.mean(x) > 0:
            x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
