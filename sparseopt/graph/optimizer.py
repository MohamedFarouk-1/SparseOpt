"""
Graph optimization module for SparseOpt.
"""

import torch
import torch.fx as fx
from typing import List, Optional, Tuple, Dict, Any
from .base import GraphPass

class GraphOptimizer:
    """Applies a sequence of graph optimization passes to a PyTorch model."""
    
    def __init__(self, passes: List[GraphPass]):
        """Initialize the optimizer with a list of passes.
        
        Args:
            passes: List of GraphPass objects to apply in sequence
        """
        self.passes = passes
    
    def optimize(self, model: torch.nn.Module, example_input: torch.Tensor) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Apply optimization passes to the model.
        
        Args:
            model: The PyTorch model to optimize
            example_input: Example input tensor for tracing
            
        Returns:
            Tuple of (optimized model, statistics dictionary)
        """
        # Trace the model
        try:
            traced = fx.symbolic_trace(model)
            print("\nInitial traced graph:")
            print(traced.graph)
            
            # Count initial nodes
            initial_nodes = len(list(traced.graph.nodes))
        except Exception as e:
            print(f"Failed to trace model: {e}")
            return model, {"error": str(e)}
        
        # Apply passes in sequence
        stats = {
            "initial_nodes": initial_nodes,
            "passes": {}
        }
        current = traced
        for i, pass_obj in enumerate(self.passes):
            try:
                print(f"\nApplying {pass_obj.__class__.__name__}...")
                current, pass_stats = pass_obj.apply(current, {"x": example_input})
                stats["passes"][f"pass_{i}"] = pass_stats
                print("Resulting graph:")
                print(current.graph)
            except Exception as e:
                print(f"Pass {pass_obj.__class__.__name__} failed: {e}")
                return traced, {"error": str(e)}
        
        # Count final nodes
        final_nodes = len(list(current.graph.nodes))
        stats["final_nodes"] = final_nodes
        
        # Verify correctness
        with torch.no_grad():
            original_output = model(example_input)
            optimized_output = current(example_input)
            is_correct = torch.allclose(original_output, optimized_output, rtol=1e-3, atol=1e-3)
            stats["correctness"] = is_correct
            if not is_correct:
                print("Warning: Optimization changed model outputs!")
                return traced, {"error": "Optimization changed model outputs"}
        
        return current, stats 