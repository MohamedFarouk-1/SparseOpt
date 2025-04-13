"""
Conv-BN-ReLU Fusion Pass.

This pass fuses Conv2d -> BatchNorm2d -> ReLU patterns into a single operation
for improved performance.
"""

import torch
import torch.nn as nn
import torch.fx as fx
from typing import Dict, Any, Tuple, List, Set
from ..base import GraphPass

class ConvBatchNormReLU(nn.Module):
    """
    Fused Conv2d-BatchNorm2d-ReLU module.
    """
    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d):
        super().__init__()
        # Create a new Conv2d with the same parameters
        self.conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None
        )
        
        # Copy the weights and bias
        self.conv.weight.data = conv.weight.data.clone()
        if conv.bias is not None:
            self.conv.bias.data = conv.bias.data.clone()
            
        # Fuse the BatchNorm parameters
        with torch.no_grad():
            # Get BatchNorm parameters
            bn_weight = bn.weight.data.clone()
            bn_bias = bn.bias.data.clone()
            bn_mean = bn.running_mean.data.clone()
            bn_var = bn.running_var.data.clone()
            bn_eps = bn.eps
            
            # Fuse the parameters
            scale = bn_weight / torch.sqrt(bn_var + bn_eps)
            self.conv.weight.data = self.conv.weight.data * scale.view(-1, 1, 1, 1)
            if self.conv.bias is not None:
                self.conv.bias.data = (self.conv.bias.data - bn_mean) * scale + bn_bias
            else:
                self.conv.bias = nn.Parameter(-bn_mean * scale + bn_bias)
                
        # Move to the same device as the input modules
        self.to(conv.weight.device)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return torch.relu(x)

class ConvBatchNormReLUFusion(GraphPass):
    def apply(self, graph_module: fx.GraphModule, example_inputs: Tuple[Any, ...]) -> Tuple[fx.GraphModule, Dict[str, Any]]:
        """
        Apply Conv-BN-ReLU fusion to the graph.
        
        Args:
            graph_module: The graph module to optimize
            example_inputs: Example inputs for tracing
            
        Returns:
            Tuple of (optimized graph module, statistics)
        """
        # Create a new graph
        new_graph = fx.Graph()
        
        # Track processed nodes and their new versions
        processed_nodes: Dict[fx.Node, fx.Node] = {}
        
        # Track statistics
        stats = {"fused_patterns": 0}
        
        # Process nodes in topological order
        for node in graph_module.graph.nodes:
            # Skip nodes that have already been processed
            if node in processed_nodes:
                continue
                
            # Look for Conv-BN-ReLU pattern
            if self._is_conv_bn_relu_pattern(node, graph_module):
                # Get the nodes in the pattern
                conv_node = node
                bn_node = list(conv_node.users)[0]
                relu_node = list(bn_node.users)[0]
                
                # Create fused module
                conv_module = graph_module.get_submodule(conv_node.target)
                bn_module = graph_module.get_submodule(bn_node.target)
                fused_module = ConvBatchNormReLU(conv_module, bn_module)
                
                # Add the fused module to the graph module
                fused_name = f"{conv_node.target}_fused"
                graph_module.add_module(fused_name, fused_module)
                
                # Create new node with fused module
                new_node = new_graph.call_module(
                    fused_name,
                    args=self._copy_args(conv_node.args, processed_nodes),
                    kwargs=conv_node.kwargs
                )
                
                # Update processed nodes
                processed_nodes[conv_node] = new_node
                processed_nodes[bn_node] = new_node
                processed_nodes[relu_node] = new_node
                
                # Update statistics
                stats["fused_patterns"] += 1
                
            else:
                # Copy the node as is
                if node.op == "placeholder":
                    new_node = new_graph.placeholder(node.name, type_expr=node.type)
                elif node.op == "output":
                    new_node = new_graph.output(self._copy_args(node.args, processed_nodes)[0])
                elif node.op == "call_module":
                    new_node = new_graph.call_module(
                        node.target,
                        args=self._copy_args(node.args, processed_nodes),
                        kwargs=node.kwargs
                    )
                elif node.op == "call_function":
                    new_node = new_graph.call_function(
                        node.target,
                        args=self._copy_args(node.args, processed_nodes),
                        kwargs=node.kwargs
                    )
                else:
                    # Copy any other type of node
                    new_node = new_graph.node_copy(node, lambda x: processed_nodes[x])
                processed_nodes[node] = new_node
                
        # Create new graph module
        new_graph_module = fx.GraphModule(graph_module, new_graph)
        
        return new_graph_module, stats
        
    def _is_conv_bn_relu_pattern(self, node: fx.Node, graph_module: fx.GraphModule) -> bool:
        """
        Check if a node is part of a Conv-BN-ReLU pattern.
        
        Args:
            node: The node to check
            graph_module: The graph module containing the node
            
        Returns:
            True if the node is part of a Conv-BN-ReLU pattern, False otherwise
        """
        # Check if node is a Conv2d module
        if not (node.op == "call_module" and 
                isinstance(graph_module.get_submodule(node.target), nn.Conv2d)):
            return False
            
        # Check if Conv2d has exactly one user
        if len(node.users) != 1:
            return False
            
        # Get the next node
        next_node = list(node.users)[0]
        
        # Check if next node is a BatchNorm2d module
        if not (next_node.op == "call_module" and 
                isinstance(graph_module.get_submodule(next_node.target), nn.BatchNorm2d)):
            return False
            
        # Check if BatchNorm2d has exactly one user
        if len(next_node.users) != 1:
            return False
            
        # Get the next node
        next_next_node = list(next_node.users)[0]
        
        # Check if next node is a ReLU module or function
        is_relu = False
        if next_next_node.op == "call_module":
            is_relu = isinstance(graph_module.get_submodule(next_next_node.target), nn.ReLU)
        elif next_next_node.op == "call_function":
            is_relu = next_next_node.target == torch.relu
            
        return is_relu
        
    def _copy_args(self, args: Tuple[Any, ...], processed_nodes: Dict[fx.Node, fx.Node]) -> Tuple[Any, ...]:
        """Helper method to copy node arguments."""
        new_args = []
        for arg in args:
            if isinstance(arg, fx.Node):
                new_args.append(processed_nodes[arg])
            else:
                new_args.append(arg)
        return tuple(new_args) 