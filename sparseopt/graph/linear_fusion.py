import torch
import torch.fx as fx
from typing import Dict, Any, Tuple, Optional
from ..modules.linear_gelu import LinearGELU
from .optimizer import GraphPass

class LinearGELUFusion(GraphPass):
    """
    FX graph-based implementation of Linear-GELU fusion.
    Detects and fuses patterns of nn.Linear followed by F.gelu into a single LinearGELU module.
    """
    
    def __init__(self):
        self.fused_count = 0
        self.fusion_stats = []
    
    def _is_linear_gelu_pattern(self, 
                               node: fx.Node,
                               graph: fx.Graph) -> Optional[Tuple[fx.Node, fx.Node]]:
        """
        Check if a node is part of a Linear-GELU pattern.
        
        Args:
            node: The node to check
            graph: The FX graph
            
        Returns:
            Tuple of (linear_node, gelu_node) if pattern found, None otherwise
        """
        # Check if node is GELU
        if not (node.op == 'call_function' and 
                node.target == torch.nn.functional.gelu and 
                len(node.args) > 0 and
                isinstance(node.args[0], fx.Node)):
            return None
            
        # Get input to GELU
        input_node = node.args[0]
        
        # Check if input is Linear
        if not (input_node.op == 'call_module' and
                isinstance(input_node.target, str)):
            return None
            
        # Get the actual module
        try:
            linear = self.graph_module.get_submodule(input_node.target)
            
            # Verify module type
            if not isinstance(linear, torch.nn.Linear):
                return None
                
        except Exception:
            return None
            
        return input_node, node
    
    def _fuse_linear_gelu(self,
                         graph: fx.Graph,
                         linear_node: fx.Node,
                         gelu_node: fx.Node) -> None:
        """
        Fuse Linear-GELU pattern into a single fused operation.
        
        Args:
            graph: The FX graph
            linear_node: The linear node
            gelu_node: The GELU node
        """
        # Get the original linear module
        linear = self.graph_module.get_submodule(linear_node.target)
        
        # Create fused module
        fused = LinearGELU(linear)
        
        # Add fused module to graph module
        fused_name = f"fused_{linear_node.target.replace('.', '_')}_gelu"
        self.graph_module.add_module(fused_name, fused)
        
        # Create the fused node
        with graph.inserting_before(gelu_node):
            fused_node = graph.create_node(
                op='call_module',
                target=fused_name,
                args=(linear_node.args[0],),  # Input to linear
                kwargs={}
            )
            
            # Replace GELU output with fused output
            gelu_node.replace_all_uses_with(fused_node)
            
            # Remove old nodes
            graph.erase_node(gelu_node)
            graph.erase_node(linear_node)
            
            # Record fusion stats
            self.fusion_stats.append({
                'linear': linear_node.target,
                'fused': fused_name
            })
            self.fused_count += 1
    
    def apply(self, graph_module: fx.GraphModule) -> Tuple[fx.GraphModule, Dict[str, Any]]:
        """
        Apply Linear-GELU fusion to the graph module.
        
        Args:
            graph_module: The FX GraphModule to optimize
            
        Returns:
            Tuple containing:
            - The optimized GraphModule
            - Dictionary of fusion statistics
        """
        self.graph_module = graph_module  # Store for use in other methods
        graph = graph_module.graph
        
        # Reset stats
        self.fused_count = 0
        self.fusion_stats = []
        
        # Find and fuse patterns
        for node in graph.nodes:
            pattern = self._is_linear_gelu_pattern(node, graph)
            if pattern is not None:
                linear_node, gelu_node = pattern
                self._fuse_linear_gelu(graph, linear_node, gelu_node)
        
        # Validate and recompile the graph
        graph.lint()
        graph_module.recompile()
        
        # Return stats
        stats = {
            'fused_count': self.fused_count,
            'fusion_stats': self.fusion_stats
        }
        
        return graph_module, stats 