"""Dropout Elimination Pass.

This pass removes dropout layers during inference since they are not needed
and can impact performance. It handles both nn.Dropout and functional dropout
operations.
"""

import torch
import torch.nn as nn
import torch.fx as fx
from typing import Dict, Any, Tuple, List, Set
from ..base import GraphPass

class DropoutElimination(GraphPass):
    """Pass to eliminate dropout operations during inference."""
    
    def apply(self, graph_module: fx.GraphModule, example_inputs: Tuple[Any, ...]) -> Tuple[fx.GraphModule, Dict[str, Any]]:
        """Apply dropout elimination to the graph.
        
        Args:
            graph_module: The graph module to optimize
            example_inputs: Example inputs for tracing
            
        Returns:
            Tuple of (optimized graph module, statistics)
        """
        # Create new graph
        new_graph = fx.Graph()
        
        # Track processed nodes and their new versions
        processed_nodes: Dict[fx.Node, fx.Node] = {}
        
        # Track statistics
        stats = {"eliminated_dropouts": 0}
        
        # Process nodes in topological order
        for node in graph_module.graph.nodes:
            # Skip nodes that have already been processed
            if node in processed_nodes:
                continue
                
            # Handle different node types
            if node.op == "placeholder":
                # Copy placeholder nodes
                new_node = new_graph.placeholder(node.name, type_expr=node.type)
                processed_nodes[node] = new_node
                
            elif node.op == "output":
                # Copy output nodes
                new_node = new_graph.output(self._copy_args(node.args, processed_nodes)[0])
                processed_nodes[node] = new_node
                
            elif node.op == "call_module":
                # Check if node is a dropout layer
                if self._is_dropout_module(node, graph_module):
                    # Skip dropout layer, connect input to output
                    processed_nodes[node] = processed_nodes[node.args[0]]
                    stats["eliminated_dropouts"] += 1
                else:
                    # Copy non-dropout module
                    new_node = new_graph.call_module(
                        node.target,
                        args=self._copy_args(node.args, processed_nodes),
                        kwargs=node.kwargs
                    )
                    processed_nodes[node] = new_node
                    
            elif node.op == "call_function":
                # Check if node is a dropout function
                if self._is_dropout_function(node):
                    # Skip dropout function, connect input to output
                    processed_nodes[node] = processed_nodes[node.args[0]]
                    stats["eliminated_dropouts"] += 1
                else:
                    # Copy non-dropout function
                    new_node = new_graph.call_function(
                        node.target,
                        args=self._copy_args(node.args, processed_nodes),
                        kwargs=node.kwargs
                    )
                    processed_nodes[node] = new_node
                    
            else:
                # Copy any other type of node
                new_node = new_graph.node_copy(node, lambda x: processed_nodes[x])
                processed_nodes[node] = new_node
                
        # Create new graph module
        new_graph_module = fx.GraphModule(graph_module, new_graph)
        return new_graph_module, stats
        
    def _is_dropout_module(self, node: fx.Node, graph_module: fx.GraphModule) -> bool:
        """Check if a node represents a dropout module."""
        if node.op != "call_module":
            return False
            
        module = graph_module.get_submodule(node.target)
        return isinstance(module, (
            nn.Dropout,
            nn.Dropout1d,
            nn.Dropout2d,
            nn.Dropout3d,
            nn.AlphaDropout
        ))
        
    def _is_dropout_function(self, node: fx.Node) -> bool:
        """Check if a node represents a dropout function."""
        if node.op != "call_function":
            return False
            
        return node.target in [
            torch.nn.functional.dropout,
            torch.nn.functional.alpha_dropout,
            torch.nn.functional.feature_alpha_dropout
        ]
        
    def _copy_args(self, args: Tuple[Any, ...], processed_nodes: Dict[fx.Node, fx.Node]) -> Tuple[Any, ...]:
        """Helper method to copy node arguments."""
        new_args = []
        for arg in args:
            if isinstance(arg, fx.Node):
                new_args.append(processed_nodes[arg])
            else:
                new_args.append(arg)
        return tuple(new_args) 