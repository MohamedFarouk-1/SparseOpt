"""
Dead Node Elimination Pass.

This pass removes dead nodes (nodes whose outputs are not used by any other nodes)
from the graph while preserving special nodes like placeholders and output nodes.
"""

import torch
import torch.nn as nn
import torch.fx as fx
from typing import Dict, Any, Tuple, List, Set
from ..base import GraphPass

class DeadNodeEliminationPass(GraphPass):
    """
    Pass to eliminate dead nodes from the graph.
    
    A dead node is one whose output is not used by any other node in the graph.
    This pass identifies and removes such nodes while preserving special nodes
    like placeholders, get_attr, and output nodes.
    """
    
    def apply(self, graph_module: fx.GraphModule, example_inputs: Tuple[Any, ...]) -> Tuple[fx.GraphModule, Dict[str, Any]]:
        """
        Apply dead node elimination to the graph.
        
        Args:
            graph_module: The graph module to optimize
            example_inputs: Example inputs for tracing
            
        Returns:
            Tuple of (optimized graph module, statistics)
        """
        graph = graph_module.graph
        
        # Find all live nodes
        live_nodes = self._find_live_nodes(graph)
        
        # Create new graph
        new_graph = fx.Graph()
        env = {}  # Maps old nodes to new nodes
        
        # Copy only live nodes to new graph
        for node in graph.nodes:
            if node in live_nodes:
                new_node = new_graph.node_copy(node, lambda x: env[x] if x in env else None)
                env[node] = new_node
                
        # Create new graph module
        new_graph_module = fx.GraphModule(graph_module, new_graph)
        
        # Return optimized module and stats
        stats = {
            "eliminated_nodes": sum(1 for node in graph.nodes if node not in live_nodes)
        }
        
        return new_graph_module, stats
        
    def _find_live_nodes(self, graph: fx.Graph) -> Set[fx.Node]:
        """Find all live nodes by working backwards from outputs."""
        live_nodes = set()
        
        # First mark output nodes as live
        for node in graph.nodes:
            if node.op == 'output':
                live_nodes.add(node)
                
        # Work backwards to mark all nodes that contribute to outputs
        worklist = list(live_nodes)
        while worklist:
            node = worklist.pop()
            for arg in node.all_input_nodes:
                if arg not in live_nodes:
                    live_nodes.add(arg)
                    worklist.append(arg)
                    
        return live_nodes
        
    def _copy_args(self, args: Tuple[Any, ...], processed_nodes: Dict[fx.Node, fx.Node]) -> Tuple[Any, ...]:
        """Helper method to copy node arguments."""
        new_args = []
        for arg in args:
            if isinstance(arg, fx.Node):
                new_args.append(processed_nodes[arg])
            else:
                new_args.append(arg)
        return tuple(new_args)
    
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
        
    def _is_node_used(self, node: fx.Node) -> bool:
        """
        Check if a node's output is used by any other node in the graph.
        
        Args:
            node: The node to check
            
        Returns:
            True if the node's output is used, False otherwise
        """
        # Special nodes are always considered used
        if node.op in ["placeholder", "get_attr", "output"]:
            return True
            
        # Check if the node's output is used by any other node
        return len(node.users) > 0 