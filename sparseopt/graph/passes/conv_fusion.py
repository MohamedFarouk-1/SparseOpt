"""
Convolution fusion passes for graph optimization.
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
        # Fuse conv and bn weights
        w = conv.weight
        b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        
        # Compute fused weights
        w_view = w.reshape(w.size(0), -1)
        gamma = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        w_fused = w_view * gamma.view(-1, 1)
        w_fused = w_fused.reshape(w.shape)
        
        # Compute fused bias
        b_fused = gamma * (b - bn.running_mean) + bn.bias
        
        # Store fused parameters
        self.weight = nn.Parameter(w_fused)
        self.bias = nn.Parameter(b_fused)
        
        # Store conv parameters
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding,
            self.dilation, self.groups
        )
        return nn.functional.relu(x)

class ConvBatchNormReLUFusion(GraphPass):
    """
    Pass to fuse Conv2d, BatchNorm2d, and ReLU operations.
    
    This pass identifies patterns of Conv2d followed by BatchNorm2d and ReLU
    and fuses them into a single operation for better performance.
    """
    
    def apply(self, graph_module: fx.GraphModule, example_inputs: Dict[str, torch.Tensor]) -> Tuple[fx.GraphModule, Dict[str, Any]]:
        """
        Apply Conv-BatchNorm-ReLU fusion to the graph module.
        
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
                
                # Check for Conv-BatchNorm-ReLU pattern
                if self._is_conv_bn_relu_pattern(node, graph_module.graph):
                    # Fuse the pattern
                    self._fuse_conv_bn_relu(node, new_graph, node_map, graph_module)
                    processed_nodes.add(node)
                    processed_nodes.add(list(node.users)[0])  # Add the BatchNorm node
                    processed_nodes.add(list(list(node.users)[0].users)[0])  # Add the ReLU node
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
    
    def _is_conv_bn_relu_pattern(self, node: fx.Node, graph: fx.Graph) -> bool:
        """
        Check if a node is part of a Conv-BatchNorm-ReLU pattern.
        
        Args:
            node: The node to check
            graph: The graph containing the node
            
        Returns:
            True if the node is part of a Conv-BatchNorm-ReLU pattern, False otherwise
        """
        # Check if the node is a Conv2d operation
        if not self._is_conv(node):
            return False
        
        # Check if the node has exactly one user
        if len(node.users) != 1:
            return False
        
        # Get the BatchNorm node
        bn_node = list(node.users)[0]
        
        # Check if it's a BatchNorm operation
        if not self._is_batch_norm(bn_node):
            return False
        
        # Check if the BatchNorm node has exactly one user
        if len(bn_node.users) != 1:
            return False
        
        # Get the ReLU node
        relu_node = list(bn_node.users)[0]
        
        # Check if it's a ReLU operation
        return self._is_relu(relu_node)
    
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
    
    def _is_batch_norm(self, node: fx.Node) -> bool:
        """
        Check if a node represents a BatchNorm operation.
        
        Args:
            node: The node to check
            
        Returns:
            True if the node represents a BatchNorm operation, False otherwise
        """
        if node.op == "call_module":
            return isinstance(node.target, str) and "batch_norm" in node.target.lower()
        
        if node.op == "call_function":
            return (
                node.target == torch.nn.functional.batch_norm
                or (isinstance(node.target, type(torch.nn.functional.batch_norm)) and "batch_norm" in str(node.target).lower())
            )
        
        return False
    
    def _is_conv(self, node: fx.Node) -> bool:
        """
        Check if a node represents a Conv2d operation.
        
        Args:
            node: The node to check
            
        Returns:
            True if the node represents a Conv2d operation, False otherwise
        """
        if node.op == "call_module":
            return isinstance(node.target, str) and "conv2d" in node.target.lower()
        
        if node.op == "call_function":
            return (
                node.target == torch.nn.functional.conv2d
                or (isinstance(node.target, type(torch.nn.functional.conv2d)) and "conv2d" in str(node.target).lower())
            )
        
        return False
    
    def _fuse_conv_bn_relu(self, conv_node: fx.Node, new_graph: fx.Graph, node_map: Dict[fx.Node, fx.Node], graph_module: fx.GraphModule) -> None:
        """
        Fuse a Conv-BatchNorm-ReLU pattern into a single operation.
        
        Args:
            conv_node: The Conv2d node
            new_graph: The new graph
            node_map: Mapping from old nodes to new nodes
            graph_module: The original graph module
        """
        # Get the BatchNorm and ReLU nodes
        bn_node = list(conv_node.users)[0]
        relu_node = list(bn_node.users)[0]
        
        # Map arguments to new nodes
        new_args = []
        for arg in conv_node.args:
            if isinstance(arg, fx.Node):
                new_args.append(node_map[arg])
            else:
                new_args.append(arg)
        
        # Map keyword arguments to new nodes
        new_kwargs = {}
        for k, v in conv_node.kwargs.items():
            if isinstance(v, fx.Node):
                new_kwargs[k] = node_map[v]
            else:
                new_kwargs[k] = v
        
        # Create a new ConvBatchNormReLU module
        conv_module = graph_module.get_submodule(conv_node.target)
        bn_module = graph_module.get_submodule(bn_node.target)
        conv_bn_relu = ConvBatchNormReLU(conv_module, bn_module)
        
        # Add the new module to the graph module
        new_module_name = f"{conv_node.target}_bn_relu"
        setattr(graph_module, new_module_name, conv_bn_relu)
        
        # Create the new node
        new_node = new_graph.call_module(new_module_name, tuple(new_args), new_kwargs)
        
        # Copy metadata
        new_node.meta = conv_node.meta.copy()
        
        # Update node map
        node_map[relu_node] = new_node
        node_map[bn_node] = new_node
        node_map[conv_node] = new_node
    
    def _copy_node(self, node: fx.Node, new_graph: fx.Graph, node_map: Dict[fx.Node, fx.Node]) -> None:
        """
        Copy a node to the new graph.
        
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
        if node.op == "placeholder":
            new_node = new_graph.placeholder(node.name, type_expr=node.type)
        elif node.op == "get_attr":
            new_node = new_graph.get_attr(node.target)
        elif node.op == "call_module":
            new_node = new_graph.call_module(node.target, tuple(new_args), new_kwargs)
        elif node.op == "call_function":
            new_node = new_graph.call_function(node.target, tuple(new_args), new_kwargs)
        elif node.op == "call_method":
            new_node = new_graph.call_method(node.target, tuple(new_args), new_kwargs)
        elif node.op == "output":
            new_node = new_graph.output(new_args[0] if new_args else None)
        else:
            raise ValueError(f"Unknown node op: {node.op}")
        
        # Copy metadata
        new_node.meta = node.meta.copy()
        
        # Update node map
        node_map[node] = new_node 