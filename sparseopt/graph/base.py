"""
Base classes for graph optimization.
"""

import torch
import torch.fx as fx
from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod

class GraphPass(ABC):
    """
    Base class for all graph optimization passes.
    
    Each pass should implement the `apply` method, which takes a graph module
    and returns an optimized graph module along with statistics about the optimization.
    """
    
    @abstractmethod
    def apply(self, graph_module: fx.GraphModule, example_inputs: Dict[str, torch.Tensor]) -> Tuple[fx.GraphModule, Dict[str, Any]]:
        """
        Apply the optimization pass to the graph module.
        
        Args:
            graph_module: The graph module to optimize
            example_inputs: Example inputs for the graph module
            
        Returns:
            Tuple of (optimized_graph_module, stats)
        """
        pass 