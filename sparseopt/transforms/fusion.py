import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

class ConvBatchNormReLUFusion:
    """Fusion pass for Conv-BN-ReLU patterns."""
    
    def __init__(self):
        self.fusion_stats = {
            'conv_bn_fused': 0,
            'conv_bn_relu_fused': 0
        }
        
    def _fuse_conv_bn(self, module: nn.Module) -> nn.Module:
        """Fuse Conv-BN pattern."""
        if not isinstance(module, nn.Conv2d):
            return module
            
        # Check if this conv is followed by BN
        next_module = None
        parent = None
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                next_module = child
                break
                
        if next_module is None:
            # Try to find BN in parent's children
            if hasattr(module, 'parent'):
                parent = module.parent
                if parent is not None:
                    for name, child in parent.named_children():
                        if isinstance(child, nn.BatchNorm2d):
                            next_module = child
                            break
                            
        if next_module is None:
            return module
            
        # Create fused conv
        fused_conv = nn.Conv2d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=True
        )
        
        # Copy weights and bias
        fused_conv.weight.data = module.weight.data
        if module.bias is not None:
            fused_conv.bias.data = module.bias.data
        else:
            fused_conv.bias.data = torch.zeros_like(next_module.bias.data)
            
        # Fuse weights
        with torch.no_grad():
            # Scale the weights
            scale = next_module.weight / torch.sqrt(next_module.running_var + next_module.eps)
            fused_conv.weight.data = fused_conv.weight.data * scale.view(-1, 1, 1, 1)
            
            # Adjust the bias
            if module.bias is not None:
                fused_conv.bias.data = (fused_conv.bias.data - next_module.running_mean) * scale + next_module.bias.data
            else:
                fused_conv.bias.data = (-next_module.running_mean) * scale + next_module.bias.data
                
        self.fusion_stats['conv_bn_fused'] += 1
        return fused_conv
        
    def _fuse_conv_bn_relu(self, module: nn.Module) -> nn.Module:
        """Fuse Conv-BN-ReLU pattern."""
        if not isinstance(module, nn.Conv2d):
            return module
            
        # First fuse Conv-BN
        fused_conv = self._fuse_conv_bn(module)
        if fused_conv is module:  # No fusion happened
            return module
            
        # Check if there's a ReLU after
        has_relu = False
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                has_relu = True
                break
                
        if not has_relu:
            # Try to find ReLU in parent's children
            if hasattr(module, 'parent'):
                parent = module.parent
                if parent is not None:
                    for name, child in parent.named_children():
                        if isinstance(child, nn.ReLU):
                            has_relu = True
                            break
                            
        if has_relu:
            self.fusion_stats['conv_bn_relu_fused'] += 1
            return nn.Sequential(fused_conv, nn.ReLU(inplace=True))
        return fused_conv
        
    def _process_module(self, module: nn.Module) -> nn.Module:
        """Process a single module for fusion opportunities."""
        if isinstance(module, nn.Conv2d):
            return self._fuse_conv_bn_relu(module)
        return module
        
    def _process_sequential(self, module: nn.Sequential) -> nn.Sequential:
        """Process a Sequential container for fusion opportunities."""
        new_modules = []
        i = 0
        while i < len(module):
            current = module[i]
            if isinstance(current, nn.Conv2d):
                # Try to fuse with next modules
                fused = self._fuse_conv_bn_relu(current)
                if fused is not current:  # Fusion happened
                    new_modules.append(fused)
                    i += 1  # Skip next modules as they were fused
                else:
                    new_modules.append(current)
                    i += 1
            else:
                new_modules.append(current)
                i += 1
        return nn.Sequential(*new_modules)
        
    def apply(self, model: nn.Module) -> nn.Module:
        """Apply fusion passes to the model."""
        for name, module in model.named_children():
            if isinstance(module, nn.Sequential):
                setattr(model, name, self._process_sequential(module))
            else:
                setattr(model, name, self._process_module(module))
        return model
 