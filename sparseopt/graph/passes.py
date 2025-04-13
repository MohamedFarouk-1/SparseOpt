import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from typing import Dict, Any, Tuple, List
from ..base import GraphPass

class LinearGELUFusion(GraphPass):
    """A graph pass that fuses Linear + GELU patterns into a single optimized module."""
    
    def __init__(self):
        super().__init__()
        self.fusions = []
    
    class FusedLinearGELU(nn.Module):
        """Fused Linear + GELU module for better performance."""
        def __init__(self, linear: nn.Linear):
            super().__init__()
            self.weight = linear.weight
            self.bias = linear.bias
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.linear(x, self.weight, self.bias)
            return F.gelu(x)
    
    def apply(self, graph_module: fx.GraphModule) -> Tuple[fx.GraphModule, Dict[str, Any]]:
        """Apply Linear + GELU fusion to the graph module.
        
        Args:
            graph_module: The input FX graph module to optimize
            
        Returns:
            Tuple of (optimized graph module, statistics dictionary)
        """
        graph = graph_module.graph
        modules = dict(graph_module.named_modules())
        self.fusions = []
        
        # Find Linear -> GELU patterns
        for node in list(graph.nodes):  # Create a copy of nodes to avoid modification during iteration
            if (node.op == 'call_function' and 
                (node.target == F.gelu or node.target == torch._C._nn.gelu)):
                prev = node.args[0]
                if prev.op == 'call_module' and isinstance(modules[prev.target], nn.Linear):
                    # Create fused module
                    linear = modules[prev.target]
                    fused = self.FusedLinearGELU(linear)
                    
                    # Add fused module to graph
                    fused_name = f"{prev.target}_gelu_fused"
                    graph_module.add_module(fused_name, fused)
                    
                    # Create new node for fused op
                    with graph.inserting_after(prev):
                        fused_node = graph.call_module(
                            fused_name,
                            args=prev.args,
                            kwargs=prev.kwargs
                        )
                    
                    # Replace all uses of the GELU node with the fused node
                    node.replace_all_uses_with(fused_node)
                    
                    # Remove the old nodes
                    graph.erase_node(node)
                    graph.erase_node(prev)
                    
                    # Track fusion for statistics
                    self.fusions.append({
                        'linear': prev.target,
                        'fused': fused_name
                    })
        
        graph.lint()
        graph_module.recompile()
        
        return graph_module, {
            'num_fusions': len(self.fusions),
            'fusions': self.fusions,
            'linear_gelu_fusions': len(self.fusions)  # Add this for the summary
        } 