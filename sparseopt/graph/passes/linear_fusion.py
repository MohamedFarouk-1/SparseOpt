"""
Linear fusion passes for graph optimization.
"""

import torch
import torch.nn as nn
import torch.fx as fx
import types
from typing import Dict, Any, Tuple, List, Set
from ..base import GraphPass

class LinearGELU(nn.Module):
    """
    Fused Linear-GELU module.
    """
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.data.clone())
        self.bias = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        self.to(linear.weight.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.linear(x, self.weight, self.bias)
        return nn.functional.gelu(x)

class LinearReLU(nn.Module):
    """
    Fused Linear-ReLU module.
    """
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.data.clone())
        self.bias = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        self.to(linear.weight.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.linear(x, self.weight, self.bias)
        return nn.functional.relu(x)

class LinearGELUFusion(GraphPass):
    """
    Pass to fuse Linear and GELU operations.
    
    This pass identifies patterns of Linear followed by GELU and fuses them
    into a single operation for better performance.
    """
    
    def apply(self, graph_module: fx.GraphModule, example_inputs: Tuple[Any, ...]) -> Tuple[fx.GraphModule, Dict[str, Any]]:
        """
        Apply Linear-GELU fusion to the graph module.
        
        Args:
            graph_module: The graph module to optimize
            example_inputs: Example inputs for the graph module
            
        Returns:
            Tuple of (optimized_graph_module, stats)
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
                
            # Look for Linear-GELU pattern
            if self._is_linear_gelu_pattern(node, graph_module):
                # Get the nodes in the pattern
                linear_node = node
                gelu_node = list(linear_node.users)[0]
                
                # Create fused module
                linear_module = graph_module.get_submodule(linear_node.target)
                fused_module = LinearGELU(linear_module)
                
                # Add the fused module to the graph module
                fused_name = f"{linear_node.target}_fused"
                graph_module.add_module(fused_name, fused_module)
                
                # Create new node with fused module
                new_node = new_graph.call_module(
                    fused_name,
                    args=self._copy_args(linear_node.args, processed_nodes),
                    kwargs=linear_node.kwargs
                )
                
                # Update processed nodes
                processed_nodes[linear_node] = new_node
                processed_nodes[gelu_node] = new_node
                
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
        
    def _is_linear_gelu_pattern(self, node: fx.Node, graph_module: fx.GraphModule) -> bool:
        """
        Check if a node is part of a Linear-GELU pattern.
        
        Args:
            node: The node to check
            graph_module: The graph module containing the node
            
        Returns:
            True if the node is part of a Linear-GELU pattern, False otherwise
        """
        # Check if node is a Linear operation
        if not self._is_linear(node, graph_module):
            return False
        
        # Check if node has exactly one user
        if len(node.users) != 1:
            return False
        
        # Check if the user is a GELU operation
        user = list(node.users)[0]
        return self._is_gelu(user, graph_module)
    
    def _is_gelu(self, node: fx.Node, graph_module: fx.GraphModule) -> bool:
        """
        Check if a node represents a GELU operation.
        
        Args:
            node: The node to check
            graph_module: The graph module containing the node
            
        Returns:
            True if the node represents a GELU operation, False otherwise
        """
        if node.op == "call_module":
            return isinstance(graph_module.get_submodule(node.target), nn.GELU)
        
        if node.op == "call_function":
            return (
                node.target == nn.functional.gelu
                or node.target == torch.nn.functional.gelu
                or node.target == torch.gelu
            )
        
        return False
    
    def _is_linear(self, node: fx.Node, graph_module: fx.GraphModule) -> bool:
        """
        Check if a node represents a Linear operation.
        
        Args:
            node: The node to check
            graph_module: The graph module containing the node
            
        Returns:
            True if the node represents a Linear operation, False otherwise
        """
        if node.op == "call_module":
            return isinstance(graph_module.get_submodule(node.target), nn.Linear)
        
        if node.op == "call_function":
            return (
                node.target == nn.functional.linear
                or node.target == torch.nn.functional.linear
                or node.target == torch.linear
            )
        
        return False
        
    def _copy_args(self, args: Tuple[Any, ...], processed_nodes: Dict[fx.Node, fx.Node]) -> Tuple[Any, ...]:
        """Helper method to copy node arguments."""
        new_args = []
        for arg in args:
            if isinstance(arg, fx.Node):
                new_args.append(processed_nodes[arg])
            else:
                new_args.append(arg)
        return tuple(new_args)

class LinearReLUFusion(GraphPass):
    """
    Pass to fuse Linear and ReLU operations.
    
    This pass identifies patterns of Linear followed by ReLU and fuses them
    into a single operation for better performance.
    """
    
    def apply(self, graph_module: fx.GraphModule, example_inputs: Tuple[Any, ...]) -> Tuple[fx.GraphModule, Dict[str, Any]]:
        """
        Apply Linear-ReLU fusion to the graph module.
        
        Args:
            graph_module: The graph module to optimize
            example_inputs: Example inputs for the graph module
            
        Returns:
            Tuple of (optimized_graph_module, stats)
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
                
            # Look for Linear-ReLU pattern
            if self._is_linear_relu_pattern(node, graph_module):
                # Get the nodes in the pattern
                linear_node = node
                relu_node = list(linear_node.users)[0]
                
                # Create fused module
                linear_module = graph_module.get_submodule(linear_node.target)
                fused_module = LinearReLU(linear_module)
                
                # Add the fused module to the graph module
                fused_name = f"{linear_node.target}_fused"
                graph_module.add_module(fused_name, fused_module)
                
                # Create new node with fused module
                new_node = new_graph.call_module(
                    fused_name,
                    args=self._copy_args(linear_node.args, processed_nodes),
                    kwargs=linear_node.kwargs
                )
                
                # Update processed nodes
                processed_nodes[linear_node] = new_node
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
        
    def _is_linear_relu_pattern(self, node: fx.Node, graph_module: fx.GraphModule) -> bool:
        """
        Check if a node is part of a Linear-ReLU pattern.
        
        Args:
            node: The node to check
            graph_module: The graph module containing the node
            
        Returns:
            True if the node is part of a Linear-ReLU pattern, False otherwise
        """
        # Check if node is a Linear operation
        if not self._is_linear(node, graph_module):
            return False
        
        # Check if node has exactly one user
        if len(node.users) != 1:
            return False
        
        # Check if the user is a ReLU operation
        user = list(node.users)[0]
        return self._is_relu(user, graph_module)
    
    def _is_relu(self, node: fx.Node, graph_module: fx.GraphModule) -> bool:
        """
        Check if a node represents a ReLU operation.
        
        Args:
            node: The node to check
            graph_module: The graph module containing the node
            
        Returns:
            True if the node represents a ReLU operation, False otherwise
        """
        if node.op == "call_module":
            return isinstance(graph_module.get_submodule(node.target), nn.ReLU)
        
        if node.op == "call_function":
            return (
                node.target == nn.functional.relu
                or node.target == torch.nn.functional.relu
                or node.target == torch.relu
            )
        
        return False
    
    def _is_linear(self, node: fx.Node, graph_module: fx.GraphModule) -> bool:
        """
        Check if a node represents a Linear operation.
        
        Args:
            node: The node to check
            graph_module: The graph module containing the node
            
        Returns:
            True if the node represents a Linear operation, False otherwise
        """
        if node.op == "call_module":
            return isinstance(graph_module.get_submodule(node.target), nn.Linear)
        
        if node.op == "call_function":
            return (
                node.target == nn.functional.linear
                or node.target == torch.nn.functional.linear
                or node.target == torch.linear
            )
        
        return False
        
    def _copy_args(self, args: Tuple[Any, ...], processed_nodes: Dict[fx.Node, fx.Node]) -> Tuple[Any, ...]:
        """Helper method to copy node arguments."""
        new_args = []
        for arg in args:
            if isinstance(arg, fx.Node):
                new_args.append(processed_nodes[arg])
            else:
                new_args.append(arg)
        return tuple(new_args) 