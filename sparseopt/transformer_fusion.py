"""Transformer-specific fusion passes for SparseOpt."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.fx import GraphModule, Node
from rich.console import Console

console = Console()

class TransformerFusionPass:
    """Base class for transformer fusion passes."""
    
    def __init__(self, name: str):
        self.name = name
        self.fusions: List[Tuple[Node, Node]] = []
    
    def match_pattern(self, node: Node) -> Optional[Node]:
        """Match the fusion pattern starting from this node.
        
        Args:
            node: The starting node to match from
            
        Returns:
            The second node in the fusion pattern if matched, None otherwise
        """
        raise NotImplementedError
    
    def apply_fusion(self, graph: torch.fx.Graph, node1: Node, node2: Node) -> None:
        """Apply the fusion pattern to the graph.
        
        Args:
            graph: The FX graph to modify
            node1: First node in the fusion pattern
            node2: Second node in the fusion pattern
        """
        raise NotImplementedError
    
    def __call__(self, graph: torch.fx.Graph) -> List[Tuple[Node, Node]]:
        """Apply the fusion pass to the graph.
        
        Args:
            graph: The FX graph to optimize
            
        Returns:
            List of (node1, node2) pairs that were fused
        """
        self.fusions = []
        
        # Find all potential fusion patterns
        for node in graph.nodes:
            if node.op == "call_module":
                fused_node = self.match_pattern(node)
                if fused_node is not None:
                    self.fusions.append((node, fused_node))
        
        # Apply fusions in reverse order to avoid invalidating node references
        for node1, node2 in reversed(self.fusions):
            self.apply_fusion(graph, node1, node2)
        
        return self.fusions

class LinearGELUFusion(TransformerFusionPass):
    """Fusion pass for Linear + GELU patterns."""
    
    def __init__(self):
        super().__init__("Linear + GELU")
    
    def match_pattern(self, node: Node) -> Optional[Node]:
        if not isinstance(node.target, str):
            return None
            
        # Check if this is a Linear layer
        if not node.target.endswith(".weight"):
            return None
            
        # Get the Linear module
        linear_module = node.graph.owning_module.get_submodule(node.target[:-7])  # Remove .weight
        if not isinstance(linear_module, nn.Linear):
            return None
            
        # Check if next node is GELU
        next_node = None
        for user in node.users:
            if user.op == "call_module" and isinstance(
                user.graph.owning_module.get_submodule(user.target),
                nn.GELU
            ):
                next_node = user
                break
                
        return next_node
    
    def apply_fusion(self, graph: torch.fx.Graph, node1: Node, node2: Node) -> None:
        # Create fused LinearGELU module
        linear_module = graph.owning_module.get_submodule(node1.target[:-7])
        gelu_module = graph.owning_module.get_submodule(node2.target)
        
        fused_module = nn.Sequential(linear_module, gelu_module)
        
        # Replace the two nodes with a single call to the fused module
        with graph.inserting_after(node2):
            new_node = graph.create_node(
                "call_module",
                target=f"{node1.target[:-7]}_gelu",
                args=(node1.args[0],),
                kwargs={}
            )
            
        # Update module references
        graph.owning_module.add_module(f"{node1.target[:-7]}_gelu", fused_module)
        
        # Replace uses of the GELU output with the fused output
        node2.replace_all_uses_with(new_node)
        
        # Remove the old nodes
        graph.erase_node(node2)
        graph.erase_node(node1)

class LinearLayerNormFusion(TransformerFusionPass):
    """Fusion pass for Linear + LayerNorm patterns."""
    
    def __init__(self):
        super().__init__("Linear + LayerNorm")
    
    def match_pattern(self, node: Node) -> Optional[Node]:
        if not isinstance(node.target, str):
            return None
            
        # Check if this is a Linear layer
        if not node.target.endswith(".weight"):
            return None
            
        # Get the Linear module
        linear_module = node.graph.owning_module.get_submodule(node.target[:-7])
        if not isinstance(linear_module, nn.Linear):
            return None
            
        # Check if next node is LayerNorm
        next_node = None
        for user in node.users:
            if user.op == "call_module" and isinstance(
                user.graph.owning_module.get_submodule(user.target),
                nn.LayerNorm
            ):
                next_node = user
                break
                
        return next_node
    
    def apply_fusion(self, graph: torch.fx.Graph, node1: Node, node2: Node) -> None:
        # Create fused LinearLayerNorm module
        linear_module = graph.owning_module.get_submodule(node1.target[:-7])
        ln_module = graph.owning_module.get_submodule(node2.target)
        
        fused_module = nn.Sequential(linear_module, ln_module)
        
        # Replace the two nodes with a single call to the fused module
        with graph.inserting_after(node2):
            new_node = graph.create_node(
                "call_module",
                target=f"{node1.target[:-7]}_ln",
                args=(node1.args[0],),
                kwargs={}
            )
            
        # Update module references
        graph.owning_module.add_module(f"{node1.target[:-7]}_ln", fused_module)
        
        # Replace uses of the LayerNorm output with the fused output
        node2.replace_all_uses_with(new_node)
        
        # Remove the old nodes
        graph.erase_node(node2)
        graph.erase_node(node1)

class MHALayerNormFusion(TransformerFusionPass):
    """Fusion pass for MultiHeadAttention + LayerNorm patterns."""
    
    def __init__(self):
        super().__init__("MultiHeadAttention + LayerNorm")
    
    def match_pattern(self, node: Node) -> Optional[Node]:
        if not isinstance(node.target, str):
            return None
            
        # Check if this is a MultiHeadAttention module
        mha_module = node.graph.owning_module.get_submodule(node.target)
        if not isinstance(mha_module, nn.MultiheadAttention):
            return None
            
        # Check if next node is LayerNorm
        next_node = None
        for user in node.users:
            if user.op == "call_module" and isinstance(
                user.graph.owning_module.get_submodule(user.target),
                nn.LayerNorm
            ):
                next_node = user
                break
                
        return next_node
    
    def apply_fusion(self, graph: torch.fx.Graph, node1: Node, node2: Node) -> None:
        # Create fused MHALayerNorm module
        mha_module = graph.owning_module.get_submodule(node1.target)
        ln_module = graph.owning_module.get_submodule(node2.target)
        
        class MHALayerNorm(nn.Module):
            def __init__(self, mha: nn.MultiheadAttention, ln: nn.LayerNorm):
                super().__init__()
                self.mha = mha
                self.ln = ln
                
            def forward(self, *args, **kwargs):
                x = self.mha(*args, **kwargs)[0]  # MHA returns (output, attention_weights)
                return self.ln(x)
        
        fused_module = MHALayerNorm(mha_module, ln_module)
        
        # Replace the two nodes with a single call to the fused module
        with graph.inserting_after(node2):
            new_node = graph.create_node(
                "call_module",
                target=f"{node1.target}_ln",
                args=node1.args,
                kwargs=node1.kwargs
            )
            
        # Update module references
        graph.owning_module.add_module(f"{node1.target}_ln", fused_module)
        
        # Replace uses of the LayerNorm output with the fused output
        node2.replace_all_uses_with(new_node)
        
        # Remove the old nodes
        graph.erase_node(node2)
        graph.erase_node(node1)

def apply_transformer_fusion_passes(graph: torch.fx.Graph) -> Dict[str, List[Tuple[Node, Node]]]:
    """Apply all transformer fusion passes to the graph.
    
    Args:
        graph: The FX graph to optimize
        
    Returns:
        Dictionary mapping fusion pass names to lists of fused node pairs
    """
    fusion_passes = [
        LinearGELUFusion(),
        LinearLayerNormFusion(),
        MHALayerNormFusion()
    ]
    
    results = {}
    for pass_ in fusion_passes:
        fusions = pass_(graph)
        if fusions:
            results[pass_.name] = fusions
            console.print(f"[green]Applied {pass_.name} fusion: {len(fusions)} patterns[/green]")
    
    return results 