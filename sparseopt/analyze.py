"""Graph analysis utilities for SparseOpt."""

import torch
from torch.fx import GraphModule, Tracer
from typing import Dict, List, Optional, Tuple
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
import networkx as nx
import re

console = Console()

class SymbolicConditionalError(Exception):
    """Exception raised when a model contains symbolic tensor conditionals."""
    pass

class ModelAnalyzer:
    """Analyzer for PyTorch models using Torch.fx."""
    
    def __init__(self, model: torch.nn.Module):
        """Initialize the analyzer with a PyTorch model.
        
        Args:
            model: PyTorch model to analyze
        """
        self.model = model
        self.tracer = Tracer()
        self.graph = None
        self.graph_module = None
        
    def trace(self) -> None:
        """Trace the model to get its computation graph."""
        try:
            self.graph = self.tracer.trace(self.model)
            self.graph_module = GraphModule(self.tracer.root, self.graph)
        except Exception as e:
            error_msg = str(e)
            
            # Check for common symbolic conditional patterns
            if "symbolic" in error_msg.lower() or "tensor" in error_msg.lower():
                # Look for specific patterns that indicate symbolic conditionals
                if re.search(r"if.*tensor.*>|<|==|!=|>=|<=", error_msg, re.IGNORECASE):
                    raise SymbolicConditionalError(
                        "Torch.fx cannot trace conditionals based on tensor values "
                        "(e.g., `if torch.mean(x) > 0`). Refactor your model to avoid "
                        "symbolic conditionals."
                    )
            
            # Re-raise the original exception if it's not a symbolic conditional
            raise
        
    def get_node_info(self, node: torch.fx.Node) -> Dict:
        """Get information about a graph node.
        
        Args:
            node: Torch.fx node to analyze
            
        Returns:
            Dictionary containing node information
        """
        return {
            "op": node.op,
            "target": str(node.target),
            "args": str(node.args),
            "kwargs": str(node.kwargs),
            "users": len(node.users),
        }
        
    def print_graph_summary(self) -> None:
        """Print a summary of the computation graph."""
        if self.graph is None:
            self.trace()
            
        table = Table(title="Graph Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        # Count different types of operations
        op_counts = {}
        for node in self.graph.nodes:
            op_counts[node.op] = op_counts.get(node.op, 0) + 1
            
        table.add_row("Total Nodes", str(len(list(self.graph.nodes))))
        for op, count in op_counts.items():
            table.add_row(f"{op} Operations", str(count))
            
        console.print(table)
        
    def print_node_details(self) -> None:
        """Print detailed information about each node in the graph."""
        if self.graph is None:
            self.trace()
            
        tree = Tree("Computation Graph")
        
        for node in self.graph.nodes:
            info = self.get_node_info(node)
            node_tree = tree.add(f"[cyan]{node.op}[/cyan] - [magenta]{node.target}[/magenta]")
            node_tree.add(f"Args: {info['args']}")
            node_tree.add(f"Kwargs: {info['kwargs']}")
            node_tree.add(f"Users: {info['users']}")
            
        console.print(tree)
        
    def get_graph_metrics(self) -> Dict:
        """Calculate various metrics about the computation graph.
        
        Returns:
            Dictionary containing graph metrics
        """
        if self.graph is None:
            self.trace()
            
        # Convert to NetworkX graph for analysis
        G = nx.DiGraph()
        for node in self.graph.nodes:
            G.add_node(node.name, **self.get_node_info(node))
            for user in node.users:
                G.add_edge(node.name, user.name)
                
        return {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
            "is_dag": nx.is_directed_acyclic_graph(G),
            "num_components": nx.number_weakly_connected_components(G),
        } 