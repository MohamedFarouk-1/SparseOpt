"""
Linear-ReLU fusion pass for graph optimization.
"""

import torch
import torch.nn as nn
import torch.fx as fx
import types
from typing import Dict, Any, Tuple, List, Set
from ..optimizer import GraphPass


class LinearReLUFusion(GraphPass):
    """
    Pass to fuse Linear and ReLU operations.
    
    This pass identifies patterns of Linear followed by ReLU and fuses them
    into a single operation for better performance.
    """
    
    def apply(self, graph_module: fx.GraphModule, example_inputs: Dict[str, torch.Tensor]) -> Tuple[fx.GraphModule, Dict[str, Any]]:
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
        
        # Track nodes that have been processed
        processed_nodes = set()
        
        # Track nodes that are ready to be processed (all inputs processed)
        ready_nodes = set()
        
        # Track fused patterns
        fused_patterns = 0
        
        # Track if we've handled the output node
        output_handled = False
        
        # Initialize ready_nodes with nodes that have no inputs
        for node in graph_module.graph.nodes:
            if node.op in ["placeholder", "get_attr"]:
                ready_nodes.add(node)
        
        # Create a mapping from old nodes to new nodes
        node_map = {}
        
        # Copy placeholder and get_attr nodes
        for node in ready_nodes:
            if node.op == "placeholder":
                new_node = new_graph.placeholder(node.name, type_expr=node.type)
                node_map[node] = new_node
                processed_nodes.add(node)
            elif node.op == "get_attr":
                new_node = new_graph.get_attr(node.target)
                node_map[node] = new_node
                processed_nodes.add(node)
        
        # Clear ready_nodes and add nodes that are now ready
        ready_nodes.clear()
        for node in graph_module.graph.nodes:
            if node not in processed_nodes and all(arg in processed_nodes or not isinstance(arg, fx.Node) for arg in node.args):
                ready_nodes.add(node)
        
        # Process nodes in topological order
        while ready_nodes:
            # Process all ready nodes
            for node in ready_nodes.copy():
                # Skip output node if already handled
                if node.op == "output" and output_handled:
                    ready_nodes.remove(node)
                    continue
                
                # Check for Linear-ReLU pattern
                if self._is_linear_relu_pattern(node, graph_module.graph):
                    # Fuse the pattern
                    self._fuse_linear_relu(node, new_graph, node_map, graph_module)
                    processed_nodes.add(node)
                    processed_nodes.add(list(node.users)[0])  # Add the ReLU node
                    fused_patterns += 1
                    ready_nodes.remove(node)
                    continue
                
                # Copy the node
                self._copy_node(node, new_graph, node_map)
                processed_nodes.add(node)
                ready_nodes.remove(node)
                
                # Mark output node as handled
                if node.op == "output":
                    output_handled = True
            
            # Update ready_nodes
            for node in graph_module.graph.nodes:
                if node not in processed_nodes and all(arg in processed_nodes or not isinstance(arg, fx.Node) for arg in node.args):
                    ready_nodes.add(node)
        
        # Create a new graph module
        new_graph_module = fx.GraphModule(graph_module, new_graph)
        
        # Return the optimized graph module and stats
        stats = {
            "fused_patterns": fused_patterns,
            "total_nodes": len(list(graph_module.graph.nodes))
        }
        
        return new_graph_module, stats
    
    def _is_linear_relu_pattern(self, node: fx.Node, graph: fx.Graph) -> bool:
        """
        Check if a node is part of a Linear-ReLU pattern.
        
        Args:
            node: The node to check
            graph: The graph containing the node
            
        Returns:
            True if the node is part of a Linear-ReLU pattern, False otherwise
        """
        # Check if the node is a Linear operation
        if not self._is_linear(node):
            return False
        
        # Check if the node has exactly one user
        if len(node.users) != 1:
            return False
        
        # Check if the user is a ReLU operation
        user = list(node.users)[0]
        return self._is_relu(user)
    
    def _is_linear(self, node: fx.Node) -> bool:
        """
        Check if a node represents a Linear operation.
        
        Args:
            node: The node to check
            
        Returns:
            True if the node represents a Linear operation, False otherwise
        """
        if node.op == "call_module":
            try:
                module = node.graph.owning_module.get_submodule(node.target)
                return isinstance(module, nn.Linear)
            except (AttributeError, ValueError):
                return False
        
        return False
    
    def _is_relu(self, node: fx.Node) -> bool:
        """
        Check if a node represents a ReLU operation.
        
        Args:
            node: The node to check
            
        Returns:
            True if the node represents a ReLU operation, False otherwise
        """
        if node.op == "call_module":
            return isinstance(node.target, str) and "relu" in node.target.lower()
        
        if node.op == "call_function":
            return (
                node.target == torch.nn.functional.relu
                or node.target == torch.relu
                or (isinstance(node.target, type(torch.relu)) and "relu" in str(node.target).lower())
            )
        
        return False
    
    def _fuse_linear_relu(self, linear_node: fx.Node, new_graph: fx.Graph, node_map: Dict[fx.Node, fx.Node], graph_module: fx.GraphModule) -> None:
        """
        Fuse a Linear-ReLU pattern into a single operation.
        
        Args:
            linear_node: The Linear node
            new_graph: The new graph
            node_map: Mapping from old nodes to new nodes
            graph_module: The original graph module
        """
        # Get the ReLU node
        relu_node = list(linear_node.users)[0]
        
        # Map arguments to new nodes
        new_args = []
        for arg in linear_node.args:
            if isinstance(arg, fx.Node):
                new_args.append(node_map[arg])
            else:
                new_args.append(arg)
        
        # Map keyword arguments to new nodes
        new_kwargs = {}
        for k, v in linear_node.kwargs.items():
            if isinstance(v, fx.Node):
                new_kwargs[k] = node_map[v]
            else:
                new_kwargs[k] = v
        
        # Create a new LinearReLU module
        linear_module = graph_module.get_submodule(linear_node.target)
        linear_relu = LinearReLU(linear_module)
        
        # Add the new module to the graph module
        new_module_name = f"{linear_node.target}_relu"
        setattr(graph_module, new_module_name, linear_relu)
        
        # Create the new node
        new_node = new_graph.call_module(new_module_name, tuple(new_args), new_kwargs)
        
        # Copy metadata
        new_node.meta = linear_node.meta.copy()
        
        # Add to node map
        node_map[linear_node] = new_node
        node_map[relu_node] = new_node
    
    def _copy_node(self, node: fx.Node, new_graph: fx.Graph, node_map: Dict[fx.Node, fx.Node]) -> None:
        """
        Copy a node from the old graph to the new graph.
        
        Args:
            node: The node to copy
            new_graph: The new graph
            node_map: Mapping from old nodes to new nodes
        """
        # Map arguments to new nodes
        new_args = []
        for arg in node.args:
            if isinstance(arg, fx.Node):
                new_args.append(node_map[arg])
            else:
                new_args.append(arg)
        
        # Map keyword arguments to new nodes
        new_kwargs = {}
        for k, v in node.kwargs.items():
            if isinstance(v, fx.Node):
                new_kwargs[k] = node_map[v]
            else:
                new_kwargs[k] = v
        
        # Create the new node
        if node.op == "call_function":
            new_node = new_graph.call_function(node.target, tuple(new_args), new_kwargs)
        elif node.op == "call_module":
            new_node = new_graph.call_module(node.target, tuple(new_args), new_kwargs)
        elif node.op == "call_method":
            new_node = new_graph.call_method(node.target, tuple(new_args), new_kwargs)
        elif node.op == "output":
            new_node = new_graph.output(tuple(new_args))
        else:
            raise ValueError(f"Unsupported node op: {node.op}")
        
        # Copy metadata
        new_node.meta = node.meta.copy()
        
        # Add to node map
        node_map[node] = new_node


class LinearReLU(nn.Module):
    """
    Fused Linear-ReLU module.
    
    This module combines a Linear layer and a ReLU activation for better performance.
    """
    
    def __init__(self, linear_module: nn.Linear):
        """
        Initialize the fused module.
        
        Args:
            linear_module: The Linear module to fuse with ReLU
        """
        super().__init__()
        self.linear = linear_module
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.relu(self.linear(x)) 