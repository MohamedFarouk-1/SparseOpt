import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearGELU(nn.Module):
    """
    A fused module that combines nn.Linear and F.gelu operations.
    This improves performance by reducing memory access and computation overhead.
    """
    
    def __init__(self, linear):
        """
        Initialize the LinearGELU module with an existing nn.Linear module.
        
        Args:
            linear: An existing nn.Linear module to fuse with GELU
        """
        super().__init__()
        self.linear = linear
        
    def forward(self, x):
        """
        Forward pass that applies linear transformation followed by GELU activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor after linear transformation and GELU activation
        """
        return F.gelu(self.linear(x)) 