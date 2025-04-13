import torch
import torch.fx as fx
from typing import Dict, Any, Tuple, List, Optional
from .optimizer import GraphPass

class ConvBnReluGraphFusion(GraphPass):
    """FX graph-based implementation of Conv-BN-ReLU fusion."""
    
    def __init__(self):
        self.fused_count = 0
        self.fusion_stats = []
    
    def _is_conv_bn_relu_pattern(self, 
                               node: fx.Node,
                               graph: fx.Graph) -> Optional[Tuple[fx.Node, fx.Node]]:
        """
        Check if a node is part of a Conv-BN-ReLU pattern.
        
        Args:
            node: The node to check
            graph: The FX graph
            
        Returns:
            Tuple of (conv_node, bn_node) if pattern found, None otherwise
        """
        # Check if node is ReLU
        if not (node.op == 'call_module' and 
                isinstance(node.target, str) and
                'relu' in node.target.lower() and 
                len(node.args) > 0 and
                isinstance(node.args[0], fx.Node)):
            return None
            
        # Get input to ReLU
        input_node = node.args[0]
        
        # Skip if input is an add operation (residual connection)
        if input_node.op == 'call_function' and input_node.target == torch.add:
            return None
            
        # Check if input is BatchNorm
        if not (input_node.op == 'call_module' and
                isinstance(input_node.target, str) and
                any(x in input_node.target.lower() for x in ['bn', 'batchnorm', 'batch_norm']) and
                len(input_node.args) > 0 and
                isinstance(input_node.args[0], fx.Node)):
            return None
            
        # Get input to BatchNorm
        conv_node = input_node.args[0]
        
        # Check if input is Conv2d
        if not (conv_node.op == 'call_module' and
                isinstance(conv_node.target, str) and
                'conv' in conv_node.target.lower()):
            return None
            
        # Get the actual modules
        try:
            conv = self.graph_module.get_submodule(conv_node.target)
            bn = self.graph_module.get_submodule(input_node.target)
            relu = self.graph_module.get_submodule(node.target)
            
            # Verify module types
            if not (isinstance(conv, torch.nn.Conv2d) and
                    isinstance(bn, torch.nn.BatchNorm2d) and
                    isinstance(relu, torch.nn.ReLU)):
                return None
                
        except Exception:
            return None
            
        return conv_node, input_node
    
    def _fuse_conv_bn_relu(self,
                          graph: fx.Graph,
                          conv_node: fx.Node,
                          bn_node: fx.Node,
                          relu_node: fx.Node) -> None:
        """
        Fuse Conv-BN-ReLU pattern into a single fused operation.
        
        Args:
            graph: The FX graph
            conv_node: The convolution node
            bn_node: The batch norm node
            relu_node: The ReLU node
        """
        # Get the original modules
        conv = self.graph_module.get_submodule(conv_node.target)
        bn = self.graph_module.get_submodule(bn_node.target)
        
        # Create a temporary Conv2d with the same parameters
        fused_conv = torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True  # We need bias for BN fusion
        )
        
        # Copy conv parameters
        fused_conv.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            fused_conv.bias.data.copy_(conv.bias.data)
        else:
            fused_conv.bias.data.zero_()
        
        # Create fused module
        fused = torch.nn.intrinsic.ConvBnReLU2d(fused_conv, bn, torch.nn.ReLU())
        
        # Add fused module to graph module
        fused_name = f"fused_{conv_node.target.replace('.', '_')}_{bn_node.target.replace('.', '_')}"
        self.graph_module.add_module(fused_name, fused)
        
        # Create the fused node
        with graph.inserting_before(relu_node):
            fused_node = graph.create_node(
                op='call_module',
                target=fused_name,
                args=(conv_node.args[0],),  # Input to conv
                kwargs={}
            )
            
            # Replace ReLU output with fused output
            relu_node.replace_all_uses_with(fused_node)
            
            # Remove old nodes
            graph.erase_node(relu_node)
            graph.erase_node(bn_node)
            graph.erase_node(conv_node)
            
            # Record fusion stats
            self.fusion_stats.append({
                'conv': conv_node.target,
                'bn': bn_node.target,
                'relu': relu_node.target,
                'fused': fused_name
            })
            self.fused_count += 1
    
    def apply(self, graph_module: fx.GraphModule) -> Tuple[fx.GraphModule, Dict[str, Any]]:
        """
        Apply Conv-BN-ReLU fusion to the graph module.
        
        Args:
            graph_module: The FX GraphModule to optimize
            
        Returns:
            Tuple containing:
            - The optimized GraphModule
            - Dictionary of fusion statistics
        """
        self.graph_module = graph_module  # Store for use in other methods
        graph = graph_module.graph
        
        # Reset stats
        self.fused_count = 0
        self.fusion_stats = []
        
        # Find and fuse patterns
        for node in graph.nodes:
            pattern = self._is_conv_bn_relu_pattern(node, graph)
            if pattern is not None:
                conv_node, bn_node = pattern
                self._fuse_conv_bn_relu(graph, conv_node, bn_node, node)
        
        # Recompile the graph
        graph_module.recompile()
        
        # Return stats
        stats = {
            'fused_count': self.fused_count,
            'fusion_stats': self.fusion_stats
        }
        
        return graph_module, stats 