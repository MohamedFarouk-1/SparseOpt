"""Optimization passes for SparseOpt."""

import torch
import torch.nn as nn
from torch.fx import GraphModule, Tracer, Node
from typing import Dict, List, Optional, Tuple, Set
from rich.console import Console
from rich.table import Table
import re
import time
import json
import os
import logging

from .analyze import SymbolicConditionalError
from .transformer_fusion import apply_transformer_fusion_passes
from .fusion_pass import FusionPass, FusionResult

console = Console()
logger = logging.getLogger(__name__)

# TODO: Future improvements
# 1. Integrate with torch.compile for better performance
# 2. Add support for advanced GNN tracing with custom message passing
# 3. Implement GPU-aware optimizations and memory profiling
# 4. Add support for sparse tensor operations
# 5. Implement more advanced graph transformations

class ModelOptimizer:
    """Handles model optimization with fusion passes."""
    
    def __init__(self):
        self.fusion_results = {}
    
    def optimize(
        self,
        model: nn.Module,
        fusion_patterns: List[FusionPass],
        layer_by_layer: bool = False,
        use_compile: bool = False
    ) -> nn.Module:
        """
        Optimize a model using the specified fusion patterns.
        
        Args:
            model: The model to optimize
            fusion_patterns: List of fusion passes to apply
            layer_by_layer: Whether to apply optimizations layer by layer
            use_compile: Whether to apply torch.compile after fusion passes
        
        Returns:
            Optimized model
        """
        # Make a copy of the model to avoid modifying the original
        optimized_model = model
        
        if layer_by_layer:
            # Apply each fusion pass to each layer separately
            for name, module in model.named_children():
                for fusion_pass in fusion_patterns:
                    try:
                        result = fusion_pass.apply(module)
                        self.fusion_results[f"{name}_{fusion_pass.name}"] = result
                    except Exception as e:
                        logger.warning(f"Failed to apply {fusion_pass.name} to {name}: {e}")
        else:
            # Apply each fusion pass to the entire model
            for fusion_pass in fusion_patterns:
                try:
                    result = fusion_pass.apply(optimized_model)
                    self.fusion_results[fusion_pass.name] = result
                except Exception as e:
                    logger.warning(f"Failed to apply {fusion_pass.name}: {e}")
        
        # Apply torch.compile if requested
        if use_compile:
            try:
                logger.info("Applying torch.compile with mode='max-autotune' and backend='inductor'")
                optimized_model = torch.compile(optimized_model, mode="max-autotune", backend="inductor")
                logger.info("Successfully applied torch.compile")
            except Exception as e:
                logger.warning(f"Failed to apply torch.compile: {e}")
        
        return optimized_model

    def trace(self) -> None:
        """Trace the model to get its computation graph.
        
        This method attempts to trace the model and provides detailed diagnostics
        for common tracing failures, especially those related to dynamic control flow
        in LLMs and MoE models.
        """
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
            
            # Check for dynamic control flow patterns common in LLMs and MoEs
            if "dynamic" in error_msg.lower() or "control flow" in error_msg.lower():
                raise SymbolicConditionalError(
                    "Torch.fx cannot trace dynamic control flow patterns. "
                    "This is common in LLMs and MoE models with conditional execution. "
                    "Consider using a layer-by-layer tracing approach instead."
                )
            
            # Check for loops in the model
            if "loop" in error_msg.lower() or "while" in error_msg.lower() or "for" in error_msg.lower():
                raise SymbolicConditionalError(
                    "Torch.fx cannot trace loops in the model. "
                    "This is common in LLMs with iterative processing. "
                    "Consider using a layer-by-layer tracing approach instead."
                )
            
            # Check for function calls that can't be traced
            if "call" in error_msg.lower() and "function" in error_msg.lower():
                raise SymbolicConditionalError(
                    "Torch.fx cannot trace certain function calls. "
                    "This is common in models with custom functions or callbacks. "
                    "Consider using a layer-by-layer tracing approach instead."
                )
            
            # Print detailed diagnostics
            console.print(f"[red]Error tracing model:[/red] {error_msg}")
            console.print("[yellow]Diagnostics:[/yellow]")
            console.print("  - This error may be due to dynamic control flow in the model")
            console.print("  - Try using layer-by-layer tracing for LLMs and MoE models")
            console.print("  - Consider using the '--skip-trace' option to skip problematic parts")
            
            # Re-raise the original exception
            raise
        
    def optimize_layer_by_layer(self, skip_layers=None, fusion_patterns=None) -> nn.Module:
        """Optimize a model layer by layer, which is useful for LLMs and MoE models.
        
        This method:
        1. Traces each layer individually
        2. Applies optimization passes to each traced layer
        3. Replaces the original layers with optimized ones
        
        Args:
            skip_layers: List of layer names to skip optimization
            fusion_patterns: List of fusion patterns to apply. If None, all patterns are applied.
                            Options: ['conv_bn_relu', 'conv_relu', 'linear_relu', 'linear_gelu', 'layer_norm']
            
        Returns:
            Optimized PyTorch model
        """
        import torch.nn as nn
        from torch.fx import symbolic_trace
        
        # Default fusion patterns if none specified
        if fusion_patterns is None:
            fusion_patterns = ['conv_bn_relu', 'conv_relu', 'linear_relu', 'linear_gelu', 'layer_norm']
        
        if skip_layers is None:
            skip_layers = []
        
        # Trace each layer
        traced_layers = self.trace_layer_by_layer(skip_layers)
        
        # Track total fusions
        total_fusions = 0
        
        # Optimize each traced layer
        for layer_name, traced_layer in traced_layers.items():
            try:
                # Debug print
                console.print(f"[dim]Optimizing layer: {layer_name}[/dim]")
                
                # Create a temporary optimizer for the layer
                layer_optimizer = ModelOptimizer(traced_layer)
                
                # Apply fusion passes to the layer
                try:
                    # Apply fusion patterns based on user selection
                    if 'conv_bn_relu' in fusion_patterns:
                        layer_optimizer._fuse_conv_bn_relu()
                    
                    if 'conv_relu' in fusion_patterns:
                        layer_optimizer._fuse_conv_relu()
                    
                    if 'linear_relu' in fusion_patterns:
                        layer_optimizer._fuse_linear_relu()
                    
                    if 'linear_gelu' in fusion_patterns:
                        layer_optimizer._fuse_linear_gelu()
                    
                    if 'layer_norm' in fusion_patterns:
                        layer_optimizer._fuse_layer_norm_patterns()
                    
                    # Get the number of fusions applied
                    layer_fusions = layer_optimizer._get_total_fusions()
                    total_fusions += layer_fusions
                    
                    # If fusions were applied, replace the original layer
                    if layer_fusions > 0:
                        # Recompile the optimized layer
                        optimized_layer = GraphModule(layer_optimizer.tracer.root, layer_optimizer.graph)
                        optimized_layer.recompile()
                        
                        # Replace the original layer with the optimized one
                        layer_path = layer_name.split('.')
                        parent_name = '.'.join(layer_path[:-1])
                        layer_name_only = layer_path[-1]
                        
                        if parent_name:
                            parent = self.model
                            for part in parent_name.split('.'):
                                parent = getattr(parent, part)
                            setattr(parent, layer_name_only, optimized_layer)
                        else:
                            setattr(self.model, layer_name_only, optimized_layer)
                        
                        console.print(f"[green]Applied {layer_fusions} fusions to layer: {layer_name}[/green]")
                    else:
                        console.print(f"[dim]No fusions applied to layer: {layer_name}[/dim]")
                except Exception as e:
                    console.print(f"[yellow]Warning:[/yellow] Failed to optimize layer {layer_name}: {str(e)}")
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Failed to process layer {layer_name}: {str(e)}")
        
        # Print optimization summary
        table = Table(title="Layer-by-Layer Optimization Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Layers", str(len(traced_layers)))
        table.add_row("Total Fusions", str(total_fusions))
        
        console.print(table)
        
        return self.model

    def trace_layer_by_layer(self, skip_layers=None) -> Dict[str, nn.Module]:
        """Trace a model layer by layer, which is useful for LLMs and MoE models.
        
        This method:
        1. Identifies all layers in the model
        2. Traces each layer individually
        3. Returns a dictionary of traced layers
        
        Args:
            skip_layers: List of layer names to skip tracing
            
        Returns:
            Dictionary mapping layer names to traced GraphModules
        """
        import torch.nn as nn
        from torch.fx import symbolic_trace
        
        if skip_layers is None:
            skip_layers = []
        
        traced_layers = {}
        skipped_layers = []
        
        # Helper function to check if a module is a layer
        def is_layer(module):
            # Common layer patterns in LLMs and MoEs
            layer_patterns = [
                "layer", "block", "transformer", "attention", "mlp", 
                "moe", "expert", "gate", "router", "ffn", "decoder", "encoder"
            ]
            
            # Check by class name
            module_class = module.__class__.__name__.lower()
            if any(pattern in module_class for pattern in layer_patterns):
                return True
            
            # Check by attribute names
            attr_names = [attr.lower() for attr in dir(module) if not attr.startswith('_')]
            if any(pattern in attr_name for attr_name in attr_names for pattern in layer_patterns):
                return True
            
            return False
        
        # Find all layers in the model
        def find_layers(module, prefix=""):
            layers = {}
            
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # Skip if this is a basic module
                if isinstance(child, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Linear, nn.LayerNorm)):
                    continue
                
                # Check if this is a layer
                if is_layer(child):
                    layers[full_name] = child
                else:
                    # Recursively search for layers in this module
                    child_layers = find_layers(child, full_name)
                    layers.update(child_layers)
            
            return layers
        
        # Find all layers
        layers = find_layers(self.model)
        
        # Trace each layer
        for layer_name, layer in layers.items():
            # Skip if this layer is in the skip list
            if layer_name in skip_layers:
                skipped_layers.append(layer_name)
                continue
            
            try:
                # Debug print
                console.print(f"[dim]Tracing layer: {layer_name}[/dim]")
                
                # Trace the layer
                traced_layer = symbolic_trace(layer)
                traced_layers[layer_name] = traced_layer
                
                console.print(f"[green]Successfully traced layer: {layer_name}[/green]")
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Failed to trace layer {layer_name}: {str(e)}")
                skipped_layers.append(layer_name)
        
        # Print summary
        table = Table(title="Layer-by-Layer Tracing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Layers", str(len(layers)))
        table.add_row("Successfully Traced", str(len(traced_layers)))
        table.add_row("Skipped Layers", str(len(skipped_layers)))
        
        console.print(table)
        
        # Print skipped layers if any
        if skipped_layers:
            console.print("\n[yellow]Skipped Layers:[/yellow]")
            for layer_name in skipped_layers:
                console.print(f"  - {layer_name}")
        
        return traced_layers

    def _fuse_linear_relu(self) -> None:
        """Fuse Linear and ReLU operations into a single module."""
        if self.graph is None:
            self.trace()
            
        # Find Linear -> ReLU patterns
        to_fuse = []
        for node in self.graph.nodes:
            if (node.op == "call_module" and 
                isinstance(node.target, str) and
                "relu" in node.target.lower()):
                
                # Check if the input to this ReLU is a Linear layer
                if (len(node.args) == 1 and 
                    isinstance(node.args[0], Node) and
                    node.args[0].op == "call_module" and
                    isinstance(node.args[0].target, str) and
                    "linear" in node.args[0].target.lower()):
                    
                    to_fuse.append((node.args[0], node))
                    
        # Create fused modules
        fused_count = 0
        for linear_node, relu_node in to_fuse:
            # Get the original modules
            linear_module = self.tracer.root.get_submodule(linear_node.target)
            relu_module = self.tracer.root.get_submodule(relu_node.target)
            
            # Create a new fused module
            class FusedLinearReLU(nn.Module):
                def __init__(self, linear, relu):
                    super().__init__()
                    self.linear = linear
                    self.relu = relu
                    
                def forward(self, x):
                    return self.relu(self.linear(x))
                    
            # Create the fused module
            fused_module = FusedLinearReLU(linear_module, relu_module)
            
            # Replace the original modules with the fused one
            parent_name = ".".join(linear_node.target.split(".")[:-1])
            module_name = linear_node.target.split(".")[-1]
            parent = self.tracer.root.get_submodule(parent_name)
            
            # Add the fused module to the parent
            setattr(parent, f"{module_name}_fused", fused_module)
            
            # Update the graph to use the fused module
            with self.graph.inserting_before(relu_node):
                new_node = self.graph.create_node(
                    "call_module",
                    f"{parent_name}.{module_name}_fused",
                    args=(linear_node.args[0],),
                    kwargs={}
                )
                
            # Replace uses of the ReLU node with the new fused node
            relu_node.replace_all_uses_with(new_node)
            
            # Remove the old nodes
            self.graph.erase_node(relu_node)
            self.graph.erase_node(linear_node)
            
            fused_count += 1
            
        # Print optimization summary
        table = Table(title="Linear-ReLU Fusion Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Fused Pairs", str(fused_count))
        
        console.print(table)
        
    def _fuse_conv_relu(self) -> None:
        """Fuse Conv2d and ReLU operations into a single module using torch.nn.intrinsic.ConvReLU2d.
        
        This function handles two patterns:
        1. nn.Conv2d → nn.ReLU module sequences
        2. nn.Conv2d → F.relu(...) functional calls
        
        It focuses on direct sequential patterns in the FX graph for better detection.
        """
        if self.graph is None:
            self.trace()
            
        # Import torch.nn.functional for pattern matching
        import torch.nn.functional as F
        
        # Find Conv2d -> ReLU patterns
        to_fuse = []
        skipped_fusions = []
        
        # Get all nodes in the graph
        nodes = list(self.graph.nodes)
        
        # Iterate through nodes to find Conv2d -> ReLU patterns
        for i, node in enumerate(nodes):
            # Skip if this is not a Conv2d module
            if not (node.op == "call_module" and 
                   isinstance(node.target, str) and
                   "conv" in node.target.lower() and
                   "2d" in node.target.lower()):
                continue
                
            # Get the Conv2d module
            try:
                conv_module = self.tracer.root.get_submodule(node.target)
                if not isinstance(conv_module, nn.Conv2d):
                    skipped_fusions.append((node, "Not a Conv2d module"))
                    continue
            except Exception as e:
                skipped_fusions.append((node, f"Failed to get Conv2d module: {str(e)}"))
                continue
                
            # Check if the next node is a ReLU (either module or functional)
            next_node = None
            fusion_type = None
            
            # Look at the next node in the graph
            if i + 1 < len(nodes):
                next_node = nodes[i + 1]
                
                # Case 1: ReLU as a module
                if (next_node.op == "call_module" and 
                    isinstance(next_node.target, str) and
                    "relu" in next_node.target.lower() and
                    next_node in node.users):  # Ensure it depends on our Conv2d
                    fusion_type = "module"
                
                # Case 2: ReLU as a functional call
                elif (next_node.op == "call_function" and 
                      next_node.target == F.relu and
                      next_node in node.users):  # Ensure it depends on our Conv2d
                    fusion_type = "functional"
            
            # If we found a match, add to fusion candidates
            if next_node and fusion_type:
                to_fuse.append((node, next_node, fusion_type))
            else:
                skipped_fusions.append((node, "No matching ReLU found in the next node"))
                
        # Create fused modules
        fused_count = 0
        for conv_node, relu_node, fusion_type in to_fuse:
            try:
                # Get the Conv2d module
                conv_module = self.tracer.root.get_submodule(conv_node.target)
                
                # Create a new fused module using torch.nn.intrinsic.ConvReLU2d
                try:
                    from torch.nn.intrinsic import ConvReLU2d
                except ImportError:
                    console.print("[yellow]Warning:[/yellow] torch.nn.intrinsic not available. Skipping Conv2d+ReLU fusion.")
                    break
                    
                # Create the fused module with the same parameters as the original Conv2d
                fused_module = ConvReLU2d(
                    conv_module.in_channels,
                    conv_module.out_channels,
                    conv_module.kernel_size,
                    conv_module.stride,
                    conv_module.padding,
                    conv_module.dilation,
                    conv_module.groups,
                    conv_module.bias is not None,
                    conv_module.padding_mode
                )
                
                # Copy the weights and bias from the original Conv2d
                fused_module[0].weight.data = conv_module.weight.data
                if conv_module.bias is not None:
                    fused_module[0].bias.data = conv_module.bias.data
                
                # Replace the original modules with the fused one
                parent_name = ".".join(conv_node.target.split(".")[:-1])
                module_name = conv_node.target.split(".")[-1]
                parent = self.tracer.root.get_submodule(parent_name)
                
                # Add the fused module to the parent
                setattr(parent, f"{module_name}_fused", fused_module)
                
                # Update the graph to use the fused module
                with self.graph.inserting_before(relu_node):
                    new_node = self.graph.create_node(
                        "call_module",
                        f"{parent_name}.{module_name}_fused",
                        args=(conv_node.args[0],),
                        kwargs={}
                    )
                    
                # Replace uses of the ReLU node with the new fused node
                relu_node.replace_all_uses_with(new_node)
                
                # Remove the old nodes
                self.graph.erase_node(relu_node)
                self.graph.erase_node(conv_node)
                
                fused_count += 1
                
            except Exception as e:
                skipped_fusions.append((conv_node, f"Fusion failed: {str(e)}"))
                
        # Print optimization summary
        table = Table(title="Conv2d-ReLU Fusion Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Fused Pairs", str(fused_count))
        table.add_row("Skipped Fusions", str(len(skipped_fusions)))
        
        console.print(table)
        
        # Print debug info for skipped fusions if any
        if skipped_fusions and console.is_terminal:
            console.print("\n[yellow]Debug: Skipped Conv2d+ReLU fusions:[/yellow]")
            for node, reason in skipped_fusions[:10]:  # Limit to first 10 to avoid cluttering output
                console.print(f"  - {node.target if hasattr(node, 'target') else node.op}: {reason}")
            if len(skipped_fusions) > 10:
                console.print(f"  - ... and {len(skipped_fusions) - 10} more")
        
    def _fuse_conv_bn_relu(self) -> None:
        """Fuse Conv2d → BatchNorm2d → ReLU operations into a single module using torch.nn.intrinsic.ConvBnReLU2d.
        
        This function handles the following pattern:
        nn.Conv2d → nn.BatchNorm2d → (nn.ReLU or F.relu)
        
        It recursively analyzes submodules to find fusion opportunities in nested structures.
        """
        if self.graph is None:
            self.trace()
            
        # Import torch.nn.functional for pattern matching
        import torch.nn.functional as F
        from torch.fx import symbolic_trace
        from torch.fx.subgraph_rewriter import replace_pattern
        
        # Track total fusions across all submodules
        total_fused_count = 0
        total_skipped_fusions = []
        submodule_fusion_counts = {}  # Track fusions by submodule name
        
        def find_conv_bn_relu_pattern(node):
            """Helper function to find Conv2d → BatchNorm2d → ReLU patterns starting from a node."""
            if not (node.op == "call_module" and 
                   isinstance(node.target, str) and
                   "conv" in node.target.lower() and
                   "2d" in node.target.lower()):
                return None
            
            # Get the Conv2d module
            try:
                conv_module = self.tracer.root.get_submodule(node.target)
                if not isinstance(conv_module, nn.Conv2d):
                    return None
            except Exception:
                return None
            
            # Find BatchNorm2d users
            bn_nodes = []
            for user in node.users:
                if (user.op == "call_module" and 
                    isinstance(user.target, str) and
                    "batchnorm" in user.target.lower() and
                    "2d" in user.target.lower()):
                    bn_nodes.append(user)
                
            if not bn_nodes:
                return None
            
            # For each BatchNorm2d, find ReLU users
            patterns = []
            for bn_node in bn_nodes:
                try:
                    bn_module = self.tracer.root.get_submodule(bn_node.target)
                    if not isinstance(bn_module, nn.BatchNorm2d):
                        continue
                except Exception:
                    continue
                
                # Find ReLU users
                for user in bn_node.users:
                    relu_type = None
                    
                    # Case 1: ReLU as a module
                    if (user.op == "call_module" and 
                        isinstance(user.target, str) and
                        "relu" in user.target.lower()):
                        relu_type = "module"
                    
                    # Case 2: ReLU as a functional call
                    elif (user.op == "call_function" and 
                          user.target == F.relu):
                        relu_type = "functional"
                    
                    if relu_type:
                        patterns.append((node, bn_node, user, relu_type))
                    
            return patterns
        
        def create_fused_module(conv_module, bn_module):
            """Helper function to create a fused ConvBnReLU2d module."""
            try:
                from torch.nn.intrinsic import ConvBnReLU2d
            except ImportError:
                console.print("[yellow]Warning:[/yellow] torch.nn.intrinsic not available. Skipping Conv2d+BatchNorm2d+ReLU fusion.")
                return None
            
            # Create the fused module with the same parameters
            fused_module = ConvBnReLU2d(
                conv_module.in_channels,
                conv_module.out_channels,
                conv_module.kernel_size,
                conv_module.stride,
                conv_module.padding,
                conv_module.dilation,
                conv_module.groups,
                conv_module.bias is not None,
                conv_module.padding_mode,
                bn_module.eps,
                bn_module.momentum
            )
            
            # Copy the weights and bias from the original Conv2d
            fused_module[0].weight.data = conv_module.weight.data
            if conv_module.bias is not None:
                fused_module[0].bias.data = conv_module.bias.data
            
            # Copy the running stats from the original BatchNorm2d
            fused_module[1].running_mean.data = bn_module.running_mean.data
            fused_module[1].running_var.data = bn_module.running_var.data
            fused_module[1].weight.data = bn_module.weight.data
            fused_module[1].bias.data = bn_module.bias.data
            
            return fused_module
        
        def process_submodule(submodule_node):
            """Helper function to process a submodule for fusion opportunities."""
            try:
                # Get the submodule instance
                submodule = self.tracer.root.get_submodule(submodule_node.target)
                
                # Skip if it's not a Module or is a basic module type
                if not isinstance(submodule, nn.Module) or isinstance(submodule, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Linear)):
                    return 0, []
                
                # Debug print
                console.print(f"[dim]Analyzing submodule: {submodule_node.target}[/dim]")
                
                # Create a new tracer for the submodule
                submodule_tracer = Tracer()
                
                # Trace the submodule
                try:
                    submodule_graph = submodule_tracer.trace(submodule)
                    submodule_graph_module = GraphModule(submodule_tracer.root, submodule_graph)
                except Exception as e:
                    console.print(f"[yellow]Warning:[/yellow] Failed to trace submodule {submodule_node.target}: {str(e)}")
                    return 0, []
                
                # Find all Conv2d → BatchNorm2d → ReLU patterns
                patterns = []
                for node in submodule_graph.nodes:
                    node_patterns = find_conv_bn_relu_pattern(node)
                    if node_patterns:
                        patterns.extend(node_patterns)
                
                # Apply fusions
                submodule_fused_count = 0
                submodule_skipped_fusions = []
                
                for conv_node, bn_node, relu_node, relu_type in patterns:
                    try:
                        # Get the modules
                        conv_module = submodule_tracer.root.get_submodule(conv_node.target)
                        bn_module = submodule_tracer.root.get_submodule(bn_node.target)
                        
                        # Create fused module
                        fused_module = create_fused_module(conv_module, bn_module)
                        if fused_module is None:
                            continue
                        
                        # Replace the original modules with the fused one
                        parent_name = ".".join(conv_node.target.split(".")[:-1])
                        module_name = conv_node.target.split(".")[-1]
                        parent = submodule_tracer.root.get_submodule(parent_name)
                        
                        # Add the fused module to the parent
                        setattr(parent, f"{module_name}_fused", fused_module)
                        
                        # Update the graph to use the fused module
                        with submodule_graph.inserting_before(relu_node):
                            new_node = submodule_graph.create_node(
                                "call_module",
                                f"{parent_name}.{module_name}_fused",
                                args=(conv_node.args[0],),
                                kwargs={}
                            )
                        
                        # Replace uses of the ReLU node with the new fused node
                        relu_node.replace_all_uses_with(new_node)
                        
                        # Remove the old nodes
                        submodule_graph.erase_node(relu_node)
                        submodule_graph.erase_node(bn_node)
                        submodule_graph.erase_node(conv_node)
                        
                        submodule_fused_count += 1
                        
                    except Exception as e:
                        submodule_skipped_fusions.append((conv_node, f"Fusion failed: {str(e)}"))
                
                # If we found and applied fusions
                if submodule_fused_count > 0:
                    # Recompile the optimized submodule
                    submodule_graph_module.recompile()
                    
                    # Replace the original submodule with the optimized one
                    parent_name = ".".join(submodule_node.target.split(".")[:-1])
                    module_name = submodule_node.target.split(".")[-1]
                    parent = self.tracer.root.get_submodule(parent_name)
                    
                    # Add the optimized submodule to the parent
                    setattr(parent, f"{module_name}_optimized", submodule_graph_module)
                    
                    # Update the main graph to use the optimized submodule
                    with self.graph.inserting_before(submodule_node):
                        new_node = self.graph.create_node(
                            "call_module",
                            f"{parent_name}.{module_name}_optimized",
                            args=submodule_node.args,
                            kwargs=submodule_node.kwargs
                        )
                    
                    # Replace uses of the original submodule node with the new one
                    submodule_node.replace_all_uses_with(new_node)
                    
                    # Remove the original submodule node
                    self.graph.erase_node(submodule_node)
                    
                    # Track fusions by submodule name
                    submodule_fusion_counts[submodule_node.target] = submodule_fused_count
                    
                    # Print submodule summary
                    console.print(f"[green]Applied {submodule_fused_count} Conv2d-BatchNorm2d-ReLU fusions in {submodule_node.target}[/green]")
                
                return submodule_fused_count, submodule_skipped_fusions
                
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Failed to process submodule {submodule_node.target}: {str(e)}")
                return 0, []
        
        # First, process all submodules
        submodules_to_process = []
        for node in self.graph.nodes:
            if (node.op == "call_module" and 
                isinstance(node.target, str) and
                "." in node.target):  # This indicates a submodule
                submodules_to_process.append(node)
        
        # Process each submodule
        for submodule_node in submodules_to_process:
            submodule_fused_count, submodule_skipped_fusions = process_submodule(submodule_node)
            total_fused_count += submodule_fused_count
            total_skipped_fusions.extend(submodule_skipped_fusions)
        
        # Now process the main graph for any remaining patterns
        patterns = []
        for node in self.graph.nodes:
            node_patterns = find_conv_bn_relu_pattern(node)
            if node_patterns:
                patterns.extend(node_patterns)
        
        # Apply fusions in the main graph
        main_fused_count = 0
        for conv_node, bn_node, relu_node, relu_type in patterns:
            try:
                # Get the modules
                conv_module = self.tracer.root.get_submodule(conv_node.target)
                bn_module = self.tracer.root.get_submodule(bn_node.target)
                
                # Create fused module
                fused_module = create_fused_module(conv_module, bn_module)
                if fused_module is None:
                    continue
                
                # Replace the original modules with the fused one
                parent_name = ".".join(conv_node.target.split(".")[:-1])
                module_name = conv_node.target.split(".")[-1]
                parent = self.tracer.root.get_submodule(parent_name)
                
                # Add the fused module to the parent
                setattr(parent, f"{module_name}_fused", fused_module)
                
                # Update the graph to use the fused module
                with self.graph.inserting_before(relu_node):
                    new_node = self.graph.create_node(
                        "call_module",
                        f"{parent_name}.{module_name}_fused",
                        args=(conv_node.args[0],),
                        kwargs={}
                    )
                
                # Replace uses of the ReLU node with the new fused node
                relu_node.replace_all_uses_with(new_node)
                
                # Remove the old nodes
                self.graph.erase_node(relu_node)
                self.graph.erase_node(bn_node)
                self.graph.erase_node(conv_node)
                
                main_fused_count += 1
                
            except Exception as e:
                total_skipped_fusions.append((conv_node, f"Fusion failed: {str(e)}"))
        
        # Update total counts
        total_fused_count += main_fused_count
        
        # Store the fused count for later use
        self._fused_count = total_fused_count
        
        # Print optimization summary
        table = Table(title="Conv2d-BatchNorm2d-ReLU Fusion Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Fused Pairs", str(total_fused_count))
        table.add_row("Main Graph Fusions", str(main_fused_count))
        table.add_row("Submodule Fusions", str(total_fused_count - main_fused_count))
        table.add_row("Total Skipped Fusions", str(len(total_skipped_fusions)))
        
        console.print(table)
        
        # Print submodule fusion details if any
        if submodule_fusion_counts:
            submodule_table = Table(title="Submodule Fusion Details")
            submodule_table.add_column("Submodule", style="cyan")
            submodule_table.add_column("Fusions Applied", style="magenta")
            
            for submodule_name, count in submodule_fusion_counts.items():
                submodule_table.add_row(submodule_name, str(count))
            
            console.print(submodule_table)
        
        # Print debug info for skipped fusions if any
        if total_skipped_fusions and console.is_terminal:
            console.print("\n[yellow]Debug: Skipped Conv2d-BatchNorm2d-ReLU fusions:[/yellow]")
            for node, reason in total_skipped_fusions[:10]:  # Limit to first 10 to avoid cluttering output
                console.print(f"  - {node.target if hasattr(node, 'target') else node.op}: {reason}")
            if len(total_skipped_fusions) > 10:
                console.print(f"  - ... and {len(total_skipped_fusions) - 10} more")
        
    def _fuse_linear_gelu(self) -> None:
        """Fuse Linear and GELU operations into a single module.
        
        This function handles the following patterns:
        1. nn.Linear → nn.GELU module sequences
        2. nn.Linear → F.gelu(...) functional calls
        
        This pattern is common in transformer-based models like GPT, BERT, etc.
        """
        if self.graph is None:
            self.trace()
            
        # Import torch.nn.functional for pattern matching
        import torch.nn.functional as F
        
        # Find Linear → GELU patterns
        to_fuse = []
        skipped_fusions = []
        
        # Get all nodes in the graph
        nodes = list(self.graph.nodes)
        
        # Iterate through nodes to find Linear → GELU patterns
        for i, node in enumerate(nodes):
            # Skip if this is not a Linear module
            if not (node.op == "call_module" and 
                   isinstance(node.target, str) and
                   "linear" in node.target.lower()):
                continue
            
            # Get the Linear module
            try:
                linear_module = self.tracer.root.get_submodule(node.target)
                if not isinstance(linear_module, nn.Linear):
                    skipped_fusions.append((node, "Not a Linear module"))
                    continue
            except Exception as e:
                skipped_fusions.append((node, f"Failed to get Linear module: {str(e)}"))
                continue
            
            # Check if the next node is a GELU (either module or functional)
            next_node = None
            fusion_type = None
            
            # Look at the next node in the graph
            if i + 1 < len(nodes):
                next_node = nodes[i + 1]
                
                # Case 1: GELU as a module
                if (next_node.op == "call_module" and 
                    isinstance(next_node.target, str) and
                    "gelu" in next_node.target.lower() and
                    next_node in node.users):  # Ensure it depends on our Linear
                    fusion_type = "module"
                
                # Case 2: GELU as a functional call
                elif (next_node.op == "call_function" and 
                      next_node.target == F.gelu and
                      next_node in node.users):  # Ensure it depends on our Linear
                    fusion_type = "functional"
            
            # If we found a match, add to fusion candidates
            if next_node and fusion_type:
                to_fuse.append((node, next_node, fusion_type))
            else:
                skipped_fusions.append((node, "No matching GELU found in the next node"))
                
        # Create fused modules
        fused_count = 0
        for linear_node, gelu_node, fusion_type in to_fuse:
            try:
                # Get the Linear module
                linear_module = self.tracer.root.get_submodule(linear_node.target)
                
                # Create a new fused module
                class FusedLinearGELU(nn.Module):
                    def __init__(self, linear, gelu=None):
                        super().__init__()
                        self.linear = linear
                        self.gelu = gelu if gelu is not None else nn.GELU()
                        
                    def forward(self, x):
                        return self.gelu(self.linear(x))
                        
                # Create the fused module
                if fusion_type == "module":
                    gelu_module = self.tracer.root.get_submodule(gelu_node.target)
                    fused_module = FusedLinearGELU(linear_module, gelu_module)
                else:
                    fused_module = FusedLinearGELU(linear_module)
                
                # Replace the original modules with the fused one
                parent_name = ".".join(linear_node.target.split(".")[:-1])
                module_name = linear_node.target.split(".")[-1]
                parent = self.tracer.root.get_submodule(parent_name)
                
                # Add the fused module to the parent
                setattr(parent, f"{module_name}_fused", fused_module)
                
                # Update the graph to use the fused module
                with self.graph.inserting_before(gelu_node):
                    new_node = self.graph.create_node(
                        "call_module",
                        f"{parent_name}.{module_name}_fused",
                        args=(linear_node.args[0],),
                        kwargs={}
                    )
                    
                # Replace uses of the GELU node with the new fused node
                gelu_node.replace_all_uses_with(new_node)
                
                # Remove the old nodes
                self.graph.erase_node(gelu_node)
                self.graph.erase_node(linear_node)
                
                fused_count += 1
                
            except Exception as e:
                skipped_fusions.append((linear_node, f"Fusion failed: {str(e)}"))
                
        # Print optimization summary
        table = Table(title="Linear-GELU Fusion Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Fused Pairs", str(fused_count))
        table.add_row("Skipped Fusions", str(len(skipped_fusions)))
        
        console.print(table)
        
        # Print debug info for skipped fusions if any
        if skipped_fusions and console.is_terminal:
            console.print("\n[yellow]Debug: Skipped Linear-GELU fusions:[/yellow]")
            for node, reason in skipped_fusions[:10]:  # Limit to first 10 to avoid cluttering output
                console.print(f"  - {node.target if hasattr(node, 'target') else node.op}: {reason}")
            if len(skipped_fusions) > 10:
                console.print(f"  - ... and {len(skipped_fusions) - 10} more")
            
        # Store the fused count for later use
        self._fused_count = fused_count

    def _fuse_layer_norm_patterns(self) -> None:
        """Fuse common LayerNorm patterns in transformer models.
        
        This function handles the following patterns:
        1. LayerNorm → Linear → LayerNorm (common in transformer blocks)
        2. LayerNorm → Dropout → LayerNorm (common in transformer blocks)
        
        These patterns are common in transformer-based models like GPT, BERT, etc.
        """
        if self.graph is None:
            self.trace()
            
        # Find LayerNorm → Linear → LayerNorm patterns
        to_fuse = []
        skipped_fusions = []
        
        # Get all nodes in the graph
        nodes = list(self.graph.nodes)
        
        # Iterate through nodes to find LayerNorm → Linear → LayerNorm patterns
        for i, node in enumerate(nodes):
            # Skip if this is not a LayerNorm module
            if not (node.op == "call_module" and 
                   isinstance(node.target, str) and
                   "layernorm" in node.target.lower()):
                continue
            
            # Get the LayerNorm module
            try:
                ln1_module = self.tracer.root.get_submodule(node.target)
                if not isinstance(ln1_module, nn.LayerNorm):
                    skipped_fusions.append((node, "Not a LayerNorm module"))
                    continue
            except Exception as e:
                skipped_fusions.append((node, f"Failed to get LayerNorm module: {str(e)}"))
                continue
            
            # Check if the next node is a Linear
            if i + 1 >= len(nodes):
                skipped_fusions.append((node, "No next node after LayerNorm"))
                continue
            
            linear_node = nodes[i + 1]
            if not (linear_node.op == "call_module" and 
                   isinstance(linear_node.target, str) and
                   "linear" in linear_node.target.lower() and
                   linear_node in node.users):  # Ensure it depends on our LayerNorm
                skipped_fusions.append((node, "Next node is not a Linear or doesn't depend on LayerNorm"))
                continue
            
            # Get the Linear module
            try:
                linear_module = self.tracer.root.get_submodule(linear_node.target)
                if not isinstance(linear_module, nn.Linear):
                    skipped_fusions.append((node, "Not a Linear module"))
                    continue
            except Exception as e:
                skipped_fusions.append((node, f"Failed to get Linear module: {str(e)}"))
                continue
            
            # Check if the next node after Linear is a LayerNorm
            if i + 2 >= len(nodes):
                skipped_fusions.append((node, "No node after Linear"))
                continue
            
            ln2_node = nodes[i + 2]
            if not (ln2_node.op == "call_module" and 
                   isinstance(ln2_node.target, str) and
                   "layernorm" in ln2_node.target.lower() and
                   ln2_node in linear_node.users):  # Ensure it depends on our Linear
                skipped_fusions.append((node, "Node after Linear is not a LayerNorm or doesn't depend on Linear"))
                continue
            
            # Get the second LayerNorm module
            try:
                ln2_module = self.tracer.root.get_submodule(ln2_node.target)
                if not isinstance(ln2_module, nn.LayerNorm):
                    skipped_fusions.append((node, "Not a LayerNorm module"))
                    continue
            except Exception as e:
                skipped_fusions.append((node, f"Failed to get LayerNorm module: {str(e)}"))
                continue
            
            # Add to fusion candidates
            to_fuse.append((node, linear_node, ln2_node))
            
        # Create fused modules
        fused_count = 0
        for ln1_node, linear_node, ln2_node in to_fuse:
            try:
                # Get the modules
                ln1_module = self.tracer.root.get_submodule(ln1_node.target)
                linear_module = self.tracer.root.get_submodule(linear_node.target)
                ln2_module = self.tracer.root.get_submodule(ln2_node.target)
                
                # Create a new fused module
                class FusedLayerNormLinearLayerNorm(nn.Module):
                    def __init__(self, ln1, linear, ln2):
                        super().__init__()
                        self.ln1 = ln1
                        self.linear = linear
                        self.ln2 = ln2
                        
                    def forward(self, x):
                        return self.ln2(self.linear(self.ln1(x)))
                        
                # Create the fused module
                fused_module = FusedLayerNormLinearLayerNorm(ln1_module, linear_module, ln2_module)
                
                # Replace the original modules with the fused one
                parent_name = ".".join(ln1_node.target.split(".")[:-1])
                module_name = ln1_node.target.split(".")[-1]
                parent = self.tracer.root.get_submodule(parent_name)
                
                # Add the fused module to the parent
                setattr(parent, f"{module_name}_fused", fused_module)
                
                # Update the graph to use the fused module
                with self.graph.inserting_before(ln2_node):
                    new_node = self.graph.create_node(
                        "call_module",
                        f"{parent_name}.{module_name}_fused",
                        args=(ln1_node.args[0],),
                        kwargs={}
                    )
                    
                # Replace uses of the second LayerNorm node with the new fused node
                ln2_node.replace_all_uses_with(new_node)
                
                # Remove the old nodes
                self.graph.erase_node(ln2_node)
                self.graph.erase_node(linear_node)
                self.graph.erase_node(ln1_node)
                
                fused_count += 1
                
            except Exception as e:
                skipped_fusions.append((ln1_node, f"Fusion failed: {str(e)}"))
                
        # Print optimization summary
        table = Table(title="LayerNorm-Linear-LayerNorm Fusion Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Fused Patterns", str(fused_count))
        table.add_row("Skipped Fusions", str(len(skipped_fusions)))
        
        console.print(table)
        
        # Print debug info for skipped fusions if any
        if skipped_fusions and console.is_terminal:
            console.print("\n[yellow]Debug: Skipped LayerNorm-Linear-LayerNorm fusions:[/yellow]")
            for node, reason in skipped_fusions[:10]:  # Limit to first 10 to avoid cluttering output
                console.print(f"  - {node.target if hasattr(node, 'target') else node.op}: {reason}")
            if len(skipped_fusions) > 10:
                console.print(f"  - ... and {len(skipped_fusions) - 10} more")
            
        # Store the fused count for later use
        self._fused_count = fused_count 

    def _print_optimization_summary(self) -> None:
        """Print a summary of the optimization results."""
        table = Table(title="Optimization Summary")
        table.add_column("Fusion Pass", style="cyan")
        table.add_column("Fused", style="green")
        table.add_column("Skipped", style="yellow")
        
        # Add CNN fusion results
        table.add_row("Conv2d → BatchNorm2d → ReLU", 
                     str(self.fusion_results["conv_bn_relu"]["fused"]), 
                     str(self.fusion_results["conv_bn_relu"]["skipped"]))
        table.add_row("Conv2d → ReLU", 
                     str(self.fusion_results["conv_relu"]["fused"]), 
                     str(self.fusion_results["conv_relu"]["skipped"]))
        
        # Add transformer fusion results
        transformer_results = self.fusion_results["transformer"]
        if isinstance(transformer_results, dict):
            for pattern, results in transformer_results.items():
                table.add_row(f"Transformer: {pattern}", 
                             str(results["fused"]), 
                             str(results["skipped"]))
        
        console.print(table)
    
    def save_report(self, path: str) -> None:
        """Save optimization results to a JSON file.
        
        Args:
            path: Path to save the report to
        """
        # Create report data
        report = {
            "model_name": self.model.__class__.__name__,
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "fusion_results": self.fusion_results
        }
        
        # Save to file
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        
        console.print(f"[green]Report saved to {path}[/green]") 