"""
Node Reordering Pass.

This pass optimizes execution by reordering nodes to group related operations
and prioritize lightweight operations.
"""

import torch
import torch.fx as fx
from typing import Dict, Any, Tuple, List, Set
from ..base import GraphPass

class NodeReorderingPass(GraphPass):
    def apply(self, graph_module: fx.GraphModule, example_inputs: Tuple[Any, ...]) -> Tuple[fx.GraphModule, Dict[str, Any]]:
        """
        Apply node reordering to the graph module.
        
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
        
        # Track nodes that have been reordered
        reordered_nodes = 0
        
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
        
        # Process nodes in batches, prioritizing lightweight operations
        while ready_nodes:
            # Group nodes by type
            lightweight_nodes = []
            compute_intensive_nodes = []
            other_nodes = []
            
            for node in ready_nodes:
                if node.op == "output":
                    processed_nodes.add(node)   # prevent re-queuing
                    continue
                if self._is_lightweight_op(node):
                    lightweight_nodes.append(node)
                elif self._is_compute_intensive_op(node):
                    compute_intensive_nodes.append(node)
                else:
                    other_nodes.append(node)
            
            # Process lightweight nodes first
            for node in lightweight_nodes:
                self._copy_node(node, new_graph, node_map)
                processed_nodes.add(node)
                reordered_nodes += 1
            
            # Then process compute-intensive nodes
            for node in compute_intensive_nodes:
                self._copy_node(node, new_graph, node_map)
                processed_nodes.add(node)
                reordered_nodes += 1
            
            # Finally process other nodes
            for node in other_nodes:
                self._copy_node(node, new_graph, node_map)
                processed_nodes.add(node)
                reordered_nodes += 1
            
            # Update ready_nodes
            ready_nodes.clear()
            for node in graph_module.graph.nodes:
                if node not in processed_nodes and all(arg in processed_nodes or not isinstance(arg, fx.Node) for arg in node.args):
                    ready_nodes.add(node)
        
        # Copy output nodes — preserve the exact return structure of the original graph
        for node in graph_module.graph.nodes:
            if node.op == "output":
                # args[0] is the value returned by the model (a single Node, a
                # tuple of Nodes, or None).  Map every embedded Node reference
                # through node_map without wrapping in an extra tuple.
                result = node.args[0] if node.args else None
                if isinstance(result, fx.Node):
                    new_result = node_map[result]
                elif isinstance(result, (tuple, list)):
                    new_result = type(result)(
                        node_map[x] if isinstance(x, fx.Node) else x
                        for x in result
                    )
                else:
                    new_result = result
                new_graph.output(new_result)
        
        # Create a new graph module
        new_graph_module = fx.GraphModule(graph_module, new_graph)
        
        # Return the optimized graph module and stats
        stats = {
            "reordered_nodes": reordered_nodes,
            "total_nodes": len(list(graph_module.graph.nodes))
        }
        
        return new_graph_module, stats
    
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
            elif isinstance(arg, (tuple, list)):
                new_args.append(type(arg)(
                    node_map[x] if isinstance(x, fx.Node) else x for x in arg
                ))
            else:
                new_args.append(arg)
        
        # Map keyword arguments to new nodes
        new_kwargs = {}
        for k, v in node.kwargs.items():
            if isinstance(v, fx.Node):
                new_kwargs[k] = node_map[v]
            elif isinstance(v, (tuple, list)):
                new_kwargs[k] = type(v)(
                    node_map[x] if isinstance(x, fx.Node) else x for x in v
                )
            else:
                new_kwargs[k] = v
        
        # Create the new node based on its operation type
        if node.op == "placeholder":
            new_node = new_graph.placeholder(node.name, type_expr=node.type)
        elif node.op == "get_attr":
            new_node = new_graph.get_attr(node.target)
        elif node.op == "call_function":
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

    def _is_lightweight_op(self, node: fx.Node) -> bool:
        """
        Determine if a node represents a lightweight operation.
        
        Args:
            node: The node to check
            
        Returns:
            True if the operation is lightweight, False otherwise
        """
        # Check for common lightweight operations
        if node.op == "call_function":
            # Basic math operations
            if node.target in [torch.add, torch.sub, torch.mul, torch.div]:
                return True
            # Activation functions
            if node.target in [torch.relu, torch.tanh, torch.sigmoid]:
                return True
            # Basic tensor operations
            if node.target in [torch.reshape, torch.transpose, torch.permute]:
                return True
                
        elif node.op == "call_module":
            # Basic layers
            if any(name in node.target for name in ["relu", "tanh", "sigmoid", "dropout"]):
                return True
                
        return False
        
    def _is_compute_intensive_op(self, node: fx.Node) -> bool:
        """
        Determine if a node represents a compute-intensive operation.
        
        Args:
            node: The node to check
            
        Returns:
            True if the operation is compute-intensive, False otherwise
        """
        # Check for common compute-intensive operations
        if node.op == "call_function":
            # Matrix operations
            if node.target in [torch.matmul, torch.mm, torch.bmm]:
                return True
            # Convolution operations
            if node.target in [torch.conv1d, torch.conv2d, torch.conv3d]:
                return True
            # Pooling operations
            if node.target in [torch.max_pool2d, torch.avg_pool2d]:
                return True
                
        elif node.op == "call_module":
            # Heavy layers
            if any(name in node.target for name in ["conv", "linear", "lstm", "gru", "transformer"]):
                return True
            # Normalization layers
            if any(name in node.target for name in ["batchnorm", "layernorm", "instancenorm"]):
                return True
                
        return False