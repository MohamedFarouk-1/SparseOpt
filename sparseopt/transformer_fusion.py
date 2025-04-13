"""Transformer-specific fusion passes for PyTorch models."""

import torch
import torch.nn as nn
from torch.fx import GraphModule, Tracer
from typing import Dict, List, Optional, Tuple, Union
from rich.console import Console
from rich.table import Table

console = Console()

class TransformerFusionPass:
    """Base class for transformer fusion passes."""
    
    def __init__(self, name: str):
        """Initialize the fusion pass.
        
        Args:
            name: Name of the fusion pass
        """
        self.name = name
        self.fused_count = 0
        self.skipped_fusions = []
    
    def apply(self, model: nn.Module) -> Tuple[nn.Module, Dict]:
        """Apply the fusion pass to the model.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Tuple of (optimized model, fusion results)
        """
        # Create a tracer
        tracer = Tracer()
        
        # Trace the model
        try:
            graph = tracer.trace(model)
            graph_module = GraphModule(tracer.root, graph)
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Failed to trace model: {str(e)}")
            return model, {"fused": 0, "skipped": 0}
        
        # Apply the fusion pass
        optimized_module = self._apply_fusion(graph_module, tracer)
        
        # Print results
        self._print_results()
        
        return optimized_module, {"fused": self.fused_count, "skipped": len(self.skipped_fusions)}
    
    def _apply_fusion(self, graph_module: GraphModule, tracer: Tracer) -> nn.Module:
        """Apply the fusion pass to the graph module.
        
        Args:
            graph_module: GraphModule to optimize
            tracer: Tracer used to trace the model
            
        Returns:
            Optimized GraphModule
        """
        raise NotImplementedError("Subclasses must implement _apply_fusion")
    
    def _print_results(self) -> None:
        """Print the results of the fusion pass."""
        table = Table(title=f"{self.name} Fusion Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Fused Patterns", str(self.fused_count))
        table.add_row("Skipped Fusions", str(len(self.skipped_fusions)))
        
        console.print(table)
        
        # Print debug info for skipped fusions if any
        if self.skipped_fusions and console.is_terminal:
            console.print(f"\n[yellow]Debug: Skipped {self.name} fusions:[/yellow]")
            for node, reason in self.skipped_fusions[:10]:  # Limit to first 10 to avoid cluttering output
                console.print(f"  - {node.target if hasattr(node, 'target') else node.op}: {reason}")
            if len(self.skipped_fusions) > 10:
                console.print(f"  - ... and {len(self.skipped_fusions) - 10} more")


class LinearLayerNormFusion(TransformerFusionPass):
    """Fusion pass for Linear → LayerNorm patterns."""
    
    def __init__(self):
        """Initialize the fusion pass."""
        super().__init__("Linear → LayerNorm")
    
    def _apply_fusion(self, graph_module: GraphModule, tracer: Tracer) -> nn.Module:
        """Apply the fusion pass to the graph module.
        
        Args:
            graph_module: GraphModule to optimize
            tracer: Tracer used to trace the model
            
        Returns:
            Optimized GraphModule
        """
        # Get all nodes in the graph
        nodes = list(graph_module.graph.nodes)
        
        # Iterate through nodes to find Linear → LayerNorm patterns
        for i, node in enumerate(nodes):
            # Skip if this is not a Linear module
            if not (node.op == "call_module" and 
                   isinstance(node.target, str) and
                   "linear" in node.target.lower()):
                continue
            
            # Get the Linear module
            try:
                linear_module = tracer.root.get_submodule(node.target)
                if not isinstance(linear_module, nn.Linear):
                    self.skipped_fusions.append((node, "Not a Linear module"))
                    continue
            except Exception as e:
                self.skipped_fusions.append((node, f"Failed to get Linear module: {str(e)}"))
                continue
            
            # Check if the next node is a LayerNorm
            if i + 1 >= len(nodes):
                self.skipped_fusions.append((node, "No next node after Linear"))
                continue
            
            ln_node = nodes[i + 1]
            if not (ln_node.op == "call_module" and 
                   isinstance(ln_node.target, str) and
                   "layernorm" in ln_node.target.lower() and
                   ln_node in node.users):  # Ensure it depends on our Linear
                self.skipped_fusions.append((node, "Next node is not a LayerNorm or doesn't depend on Linear"))
                continue
            
            # Get the LayerNorm module
            try:
                ln_module = tracer.root.get_submodule(ln_node.target)
                if not isinstance(ln_module, nn.LayerNorm):
                    self.skipped_fusions.append((node, "Not a LayerNorm module"))
                    continue
            except Exception as e:
                self.skipped_fusions.append((node, f"Failed to get LayerNorm module: {str(e)}"))
                continue
            
            # Create a new fused module
            class FusedLinearLayerNorm(nn.Module):
                def __init__(self, linear, ln):
                    super().__init__()
                    self.linear = linear
                    self.ln = ln
                    
                def forward(self, x):
                    return self.ln(self.linear(x))
                    
            # Create the fused module
            fused_module = FusedLinearLayerNorm(linear_module, ln_module)
            
            # Replace the original modules with the fused one
            parent_name = ".".join(node.target.split(".")[:-1])
            module_name = node.target.split(".")[-1]
            parent = tracer.root.get_submodule(parent_name)
            
            # Add the fused module to the parent
            setattr(parent, f"{module_name}_fused", fused_module)
            
            # Update the graph to use the fused module
            with graph_module.graph.inserting_before(ln_node):
                new_node = graph_module.graph.create_node(
                    "call_module",
                    f"{parent_name}.{module_name}_fused",
                    args=(node.args[0],),
                    kwargs={}
                )
                
            # Replace uses of the LayerNorm node with the new fused node
            ln_node.replace_all_uses_with(new_node)
            
            # Remove the old nodes
            graph_module.graph.erase_node(ln_node)
            graph_module.graph.erase_node(node)
            
            self.fused_count += 1
        
        # Recompile the graph to ensure it's valid
        graph_module.recompile()
        
        return graph_module


class LinearGELUFusion(TransformerFusionPass):
    """Fusion pass for Linear → GELU patterns."""
    
    def __init__(self):
        """Initialize the fusion pass."""
        super().__init__("Linear → GELU")
    
    def _apply_fusion(self, graph_module: GraphModule, tracer: Tracer) -> nn.Module:
        """Apply the fusion pass to the graph module.
        
        Args:
            graph_module: GraphModule to optimize
            tracer: Tracer used to trace the model
            
        Returns:
            Optimized GraphModule
        """
        # Get all nodes in the graph
        nodes = list(graph_module.graph.nodes)
        
        # Iterate through nodes to find Linear → GELU patterns
        for i, node in enumerate(nodes):
            # Skip if this is not a Linear module
            if not (node.op == "call_module" and 
                   isinstance(node.target, str) and
                   "linear" in node.target.lower()):
                continue
            
            # Get the Linear module
            try:
                linear_module = tracer.root.get_submodule(node.target)
                if not isinstance(linear_module, nn.Linear):
                    self.skipped_fusions.append((node, "Not a Linear module"))
                    continue
            except Exception as e:
                self.skipped_fusions.append((node, f"Failed to get Linear module: {str(e)}"))
                continue
            
            # Check if the next node is a GELU
            if i + 1 >= len(nodes):
                self.skipped_fusions.append((node, "No next node after Linear"))
                continue
            
            gelu_node = nodes[i + 1]
            if not (gelu_node.op == "call_module" and 
                   isinstance(gelu_node.target, str) and
                   "gelu" in gelu_node.target.lower() and
                   gelu_node in node.users):  # Ensure it depends on our Linear
                self.skipped_fusions.append((node, "Next node is not a GELU or doesn't depend on Linear"))
                continue
            
            # Get the GELU module
            try:
                gelu_module = tracer.root.get_submodule(gelu_node.target)
                if not isinstance(gelu_module, nn.GELU):
                    self.skipped_fusions.append((node, "Not a GELU module"))
                    continue
            except Exception as e:
                self.skipped_fusions.append((node, f"Failed to get GELU module: {str(e)}"))
                continue
            
            # Create a new fused module
            class FusedLinearGELU(nn.Module):
                def __init__(self, linear, gelu):
                    super().__init__()
                    self.linear = linear
                    self.gelu = gelu
                    
                def forward(self, x):
                    return self.gelu(self.linear(x))
                    
            # Create the fused module
            fused_module = FusedLinearGELU(linear_module, gelu_module)
            
            # Replace the original modules with the fused one
            parent_name = ".".join(node.target.split(".")[:-1])
            module_name = node.target.split(".")[-1]
            parent = tracer.root.get_submodule(parent_name)
            
            # Add the fused module to the parent
            setattr(parent, f"{module_name}_fused", fused_module)
            
            # Update the graph to use the fused module
            with graph_module.graph.inserting_before(gelu_node):
                new_node = graph_module.graph.create_node(
                    "call_module",
                    f"{parent_name}.{module_name}_fused",
                    args=(node.args[0],),
                    kwargs={}
                )
                
            # Replace uses of the GELU node with the new fused node
            gelu_node.replace_all_uses_with(new_node)
            
            # Remove the old nodes
            graph_module.graph.erase_node(gelu_node)
            graph_module.graph.erase_node(node)
            
            self.fused_count += 1
        
        # Recompile the graph to ensure it's valid
        graph_module.recompile()
        
        return graph_module


class MHALayerNormFusion(TransformerFusionPass):
    """Fusion pass for MultiHeadAttention → LayerNorm patterns."""
    
    def __init__(self):
        """Initialize the fusion pass."""
        super().__init__("MultiHeadAttention → LayerNorm")
    
    def _apply_fusion(self, graph_module: GraphModule, tracer: Tracer) -> nn.Module:
        """Apply the fusion pass to the graph module.
        
        Args:
            graph_module: GraphModule to optimize
            tracer: Tracer used to trace the model
            
        Returns:
            Optimized GraphModule
        """
        # Get all nodes in the graph
        nodes = list(graph_module.graph.nodes)
        
        # Iterate through nodes to find MultiHeadAttention → LayerNorm patterns
        for i, node in enumerate(nodes):
            # Skip if this is not a MultiHeadAttention module
            if not (node.op == "call_module" and 
                   isinstance(node.target, str) and
                   "attention" in node.target.lower()):
                continue
            
            # Get the MultiHeadAttention module
            try:
                mha_module = tracer.root.get_submodule(node.target)
                if not hasattr(mha_module, "forward"):  # Check if it's a callable module
                    self.skipped_fusions.append((node, "Not a callable module"))
                    continue
            except Exception as e:
                self.skipped_fusions.append((node, f"Failed to get MultiHeadAttention module: {str(e)}"))
                continue
            
            # Check if the next node is a LayerNorm
            if i + 1 >= len(nodes):
                self.skipped_fusions.append((node, "No next node after MultiHeadAttention"))
                continue
            
            ln_node = nodes[i + 1]
            if not (ln_node.op == "call_module" and 
                   isinstance(ln_node.target, str) and
                   "layernorm" in ln_node.target.lower() and
                   ln_node in node.users):  # Ensure it depends on our MultiHeadAttention
                self.skipped_fusions.append((node, "Next node is not a LayerNorm or doesn't depend on MultiHeadAttention"))
                continue
            
            # Get the LayerNorm module
            try:
                ln_module = tracer.root.get_submodule(ln_node.target)
                if not isinstance(ln_module, nn.LayerNorm):
                    self.skipped_fusions.append((node, "Not a LayerNorm module"))
                    continue
            except Exception as e:
                self.skipped_fusions.append((node, f"Failed to get LayerNorm module: {str(e)}"))
                continue
            
            # Create a new fused module
            class FusedMHALayerNorm(nn.Module):
                def __init__(self, mha, ln):
                    super().__init__()
                    self.mha = mha
                    self.ln = ln
                    
                def forward(self, *args, **kwargs):
                    return self.ln(self.mha(*args, **kwargs))
                    
            # Create the fused module
            fused_module = FusedMHALayerNorm(mha_module, ln_module)
            
            # Replace the original modules with the fused one
            parent_name = ".".join(node.target.split(".")[:-1])
            module_name = node.target.split(".")[-1]
            parent = tracer.root.get_submodule(parent_name)
            
            # Add the fused module to the parent
            setattr(parent, f"{module_name}_fused", fused_module)
            
            # Update the graph to use the fused module
            with graph_module.graph.inserting_before(ln_node):
                new_node = graph_module.graph.create_node(
                    "call_module",
                    f"{parent_name}.{module_name}_fused",
                    args=node.args,
                    kwargs=node.kwargs
                )
                
            # Replace uses of the LayerNorm node with the new fused node
            ln_node.replace_all_uses_with(new_node)
            
            # Remove the old nodes
            graph_module.graph.erase_node(ln_node)
            graph_module.graph.erase_node(node)
            
            self.fused_count += 1
        
        # Recompile the graph to ensure it's valid
        graph_module.recompile()
        
        return graph_module


def apply_transformer_fusion_passes(model: nn.Module) -> Tuple[nn.Module, Dict]:
    """Apply all transformer-specific fusion passes to the model.
    
    Args:
        model: PyTorch model to optimize
        
    Returns:
        Tuple of (optimized model, fusion results)
    """
    # Create a copy of the model to avoid modifying the original
    model = type(model)(model.config) if hasattr(model, "config") else type(model)()
    model.load_state_dict(model.state_dict())
    
    # Apply each fusion pass
    fusion_passes = [
        LinearLayerNormFusion(),
        LinearGELUFusion(),
        MHALayerNormFusion()
    ]
    
    # Track results for each pass
    results = {}
    
    for fusion_pass in fusion_passes:
        model, pass_results = fusion_pass.apply(model)
        results[fusion_pass.name] = pass_results
    
    # Print overall summary
    table = Table(title="Transformer Fusion Summary")
    table.add_column("Fusion Pass", style="cyan")
    table.add_column("Fused", style="green")
    table.add_column("Skipped", style="yellow")
    
    for pass_name, pass_results in results.items():
        table.add_row(pass_name, 
                     str(pass_results["fused"]), 
                     str(pass_results["skipped"]))
    
    console.print(table)
    
    return model, results 