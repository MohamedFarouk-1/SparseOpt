import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx import GraphModule
from torch.fx.graph import Graph, Node
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Callable
import logging
import copy
import torch.nn.utils.fusion as fuser
import torch.nn.utils.prune as prune
from torch.nn.utils.fusion import fuse_conv_bn_eval

logger = logging.getLogger(__name__)

class FusionResult:
    """Tracks the results of fusion operations."""
    
    def __init__(self):
        self.total_fused_pairs = 0
        self.main_graph_fusions = 0
        self.submodule_fusions = 0
        self.skipped_fusions = 0
        self.fusion_details = []
    
    def update(self, other: 'FusionResult'):
        """Merge another FusionResult into this one."""
        self.total_fused_pairs += other.total_fused_pairs
        self.main_graph_fusions += other.main_graph_fusions
        self.submodule_fusions += other.submodule_fusions
        self.skipped_fusions += other.skipped_fusions
        self.fusion_details.extend(other.fusion_details)
    
    def add_fusion(self, module_path: str, pattern_name: str, is_submodule: bool):
        """Record a successful fusion."""
        self.total_fused_pairs += 1
        if is_submodule:
            self.submodule_fusions += 1
        else:
            self.main_graph_fusions += 1
        self.fusion_details.append((module_path, pattern_name))
    
    def add_skipped(self):
        """Record a skipped fusion."""
        self.skipped_fusions += 1
    
    def __str__(self):
        return (
            f"Summary             \n"
            f"┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓\n"
            f"┃ Total Fused Pairs     ┃ {self.total_fused_pairs:<5} ┃\n"
            f"┃ Main Graph Fusions    ┃ {self.main_graph_fusions:<5} ┃\n"
            f"┃ Submodule Fusions     ┃ {self.submodule_fusions:<5} ┃\n"
            f"┃ Total Skipped Fusions ┃ {self.skipped_fusions:<5} ┃\n"
            f"┗━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━┛"
        )

def _get_module_qualname(gm: GraphModule, target: str) -> str:
    """Get the qualified name of a module."""
    if isinstance(target, str):
        if target.startswith("self."):
            return target[5:]  # Remove "self." prefix
        return target
    return target  # Handle cases where target is not a string

class FusionPass:
    """Base class for fusion passes."""
    
    def __init__(self, name: str):
        self.name = name
        self.fusion_results = FusionResult()
    
    def apply(self, model: torch.nn.Module) -> FusionResult:
        """Apply the fusion pass to a model recursively."""
        # Ensure model is in eval mode
        model.eval()
        
        # Apply fusions
        result = self._apply_to_module(model, "")
        
        return result
    
    def _fuse_conv_bn(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
        """Fuse Conv2d and BatchNorm2d into a single Conv2d module."""
        # Ensure both modules are in eval mode
        conv.eval()
        bn.eval()
        
        # Create copies of the modules to avoid modifying the originals
        conv_copy = copy.deepcopy(conv)
        bn_copy = copy.deepcopy(bn)
        
        # Use PyTorch's built-in fusion
        fused_conv = fuse_conv_bn_eval(conv_copy, bn_copy)
        
        return fused_conv
    
    def _apply_to_module(self, module: torch.nn.Module, path_prefix: str) -> FusionResult:
        """Apply fusion to a module and its submodules recursively."""
        result = FusionResult()
        
        # Skip basic modules that can't contain submodules
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Linear)):
            return result
        
        # Get all children first
        children = dict(module.named_children())
        
        # Try to fuse Conv-BN-ReLU patterns in this module
        fusions_to_apply = []
        for name, child in children.items():
            if isinstance(child, nn.Conv2d):
                # Look for BN and ReLU after Conv
                bn_name = name.replace('conv', 'bn')
                if hasattr(module, bn_name):
                    bn_module = getattr(module, bn_name)
                    if isinstance(bn_module, nn.BatchNorm2d):
                        # Found Conv-BN pattern
                        fusions_to_apply.append((name, child, bn_name, bn_module))
        
        # Apply fusions
        for name, conv, bn_name, bn in fusions_to_apply:
            try:
                # Fuse Conv and BN
                fused_conv = self._fuse_conv_bn(conv, bn)
                
                # Get the ReLU parameters if it exists
                relu_name = name.replace('conv', 'relu')
                relu_inplace = True  # Default to True for ResNet
                if hasattr(module, relu_name):
                    relu_module = getattr(module, relu_name)
                    if isinstance(relu_module, nn.ReLU):
                        relu_inplace = relu_module.inplace
                
                # Create a new sequential module with the fused conv and ReLU
                fused_module = nn.Sequential(
                    fused_conv,
                    nn.ReLU(inplace=relu_inplace)
                )
                
                # Replace the original Conv with the fused module
                setattr(module, name, fused_module)
                
                # Make the BatchNorm module a no-op instead of deleting it
                # This preserves the model's structure while making the BN do nothing
                with torch.no_grad():
                    bn.reset_parameters()  # Reset parameters to identity transform
                    bn.running_mean.zero_()  # Set running mean to 0
                    bn.running_var.fill_(1.0)  # Set running variance to 1
                    bn.weight.data.fill_(1.0)  # Set gamma to 1
                    bn.bias.data.zero_()  # Set beta to 0
                    bn.eval()  # Set to eval mode
                    bn.requires_grad_(False)  # Freeze parameters
                
                # Record the fusion
                full_path = f"{path_prefix}.{name}" if path_prefix else name
                result.add_fusion(full_path, "Conv2d-BatchNorm2d-ReLU", bool(path_prefix))
                
            except Exception as e:
                logger.error(f"Error during direct fusion: {e}", exc_info=True)
                result.add_skipped()
        
        # Apply to child modules recursively
        children = dict(module.named_children())  # Get updated list of children
        for name, child in children.items():
            child_path = f"{path_prefix}.{name}" if path_prefix else name
            child_result = self._apply_to_module(child, child_path)
            result.update(child_result)
        
        return result
    
    def _apply_to_graph_module(self, gm: GraphModule, path_prefix: str) -> FusionResult:
        """Apply fusion to a specific GraphModule."""
        result = FusionResult()
        modules = dict(gm.named_modules())
        
        # Patterns we're looking for 
        patterns = self._find_patterns(gm)
        logger.info(f"Found {len(patterns)} potential fusion patterns in {path_prefix or 'main module'}")
        
        # Fuse each pattern
        for pattern in patterns:
            try:
                if len(pattern) == 3:  # Conv-BN-ReLU
                    conv_node, bn_node, relu_node = pattern
                    conv_name = _get_module_qualname(gm, conv_node.target)
                    bn_name = _get_module_qualname(gm, bn_node.target)
                    
                    logger.info(f"Attempting to fuse {conv_name} -> {bn_name} -> ReLU")
                    
                    # Get modules
                    conv_module = modules[conv_name]
                    bn_module = modules[bn_name]
                    
                    if not isinstance(conv_module, nn.Conv2d) or not isinstance(bn_module, nn.BatchNorm2d):
                        logger.warning(f"Skipping fusion: {conv_name} or {bn_name} not expected module types")
                        result.add_skipped()
                        continue
                    
                    # Check for multiple users
                    if len(list(conv_node.users)) > 1 or len(list(bn_node.users)) > 1:
                        logger.warning(f"Skipping fusion: Multiple users for {conv_name} or {bn_name}")
                        result.add_skipped()
                        continue
                    
                    # Fuse Conv2d and BatchNorm2d
                    fused_conv = self._fuse_conv_bn(conv_module, bn_module)
                    
                    # Create a name for the fused module
                    # Replace dots with underscores to create a valid module name
                    fused_name = f"{conv_name.replace('.', '_')}_fused"
                    
                    # Register the fused module
                    gm.add_module(fused_name, fused_conv)
                    
                    # Create a new node for the fused module
                    with gm.graph.inserting_after(conv_node):
                        new_node = gm.graph.call_module(
                            fused_name,
                            args=conv_node.args,
                            kwargs=conv_node.kwargs
                        )
                    
                    # Rewire the graph: all users of the BN node should now use the fused Conv output
                    for user in list(bn_node.users):
                        user.replace_input_with(bn_node, new_node)
                    
                    # Remove old nodes from the graph (in reverse order to avoid reference errors)
                    gm.graph.erase_node(bn_node)
                    gm.graph.erase_node(conv_node)
                    
                    # Record the fusion
                    full_path = f"{path_prefix}.{conv_name}" if path_prefix else conv_name
                    result.add_fusion(full_path, "Conv2d-BatchNorm2d-ReLU", bool(path_prefix))
                    
                elif len(pattern) == 2:  # Conv-ReLU or Linear-ReLU
                    first_node, relu_node = pattern
                    first_name = _get_module_qualname(gm, first_node.target)
                    
                    # For simpler patterns, we just count them but don't modify
                    # as there's no mathematical simplification like with Conv-BN
                    first_module = modules[first_name]
                    if isinstance(first_module, nn.Conv2d):
                        pattern_name = "Conv2d-ReLU"
                    elif isinstance(first_module, nn.Linear):
                        pattern_name = "Linear-ReLU"
                    else:
                        result.add_skipped()
                        continue
                    
                    full_path = f"{path_prefix}.{first_name}" if path_prefix else first_name
                    result.add_fusion(full_path, pattern_name, bool(path_prefix))
                    
            except Exception as e:
                logger.error(f"Error during fusion: {e}", exc_info=True)
                result.add_skipped()
        
        # Recompile if we made changes
        if result.total_fused_pairs > 0:
            gm.recompile()
            gm.graph.lint()  # Verify the graph is still valid
        
        return result
    
    def _find_patterns(self, gm: GraphModule) -> List[List[Node]]:
        """Find all fusion patterns in a GraphModule."""
        patterns = []
        modules = dict(gm.named_modules())
        
        # Check if we can apply a special optimizer for ResNet structure
        if self._is_resnet_model(gm):
            # Special handling for ResNet bottleneck blocks
            return self._find_resnet_patterns(gm, modules)
        
        # Standard pattern detection
        for node in gm.graph.nodes:
            if node.op == 'call_module':
                target = _get_module_qualname(gm, node.target)
                
                if target in modules and isinstance(modules[target], nn.Conv2d):
                    # Check for Conv -> BN -> ReLU pattern
                    conv_bn_relu = self._trace_conv_bn_relu(node, gm, modules)
                    if conv_bn_relu:
                        patterns.append(conv_bn_relu)
                        continue
                    
                    # Check for Conv -> ReLU pattern
                    conv_relu = self._trace_conv_relu(node, gm, modules)
                    if conv_relu:
                        patterns.append(conv_relu)
                
                # Check for Linear -> ReLU pattern
                elif target in modules and isinstance(modules[target], nn.Linear):
                    linear_relu = self._trace_linear_relu(node, gm, modules)
                    if linear_relu:
                        patterns.append(linear_relu)
        
        return patterns
    
    def _is_resnet_model(self, gm: GraphModule) -> bool:
        """Check if this looks like a ResNet model."""
        # Look for typical ResNet module patterns
        module_names = [name for name, _ in gm.named_modules()]
        resnet_indicators = ['layer1', 'layer2', 'layer3', 'layer4', 'downsample', 'BasicBlock', 'Bottleneck']
        return any(indicator in name for indicator in resnet_indicators for name in module_names)
    
    def _find_resnet_patterns(self, gm: GraphModule, modules: Dict[str, nn.Module]) -> List[List[Node]]:
        """Find patterns in ResNet-like models."""
        patterns = []
        
        # Find all Conv-BN nodes even if they have multiple users
        for node in gm.graph.nodes:
            if node.op == 'call_module':
                target = _get_module_qualname(gm, node.target)
                
                if target in modules and isinstance(modules[target], nn.Conv2d):
                    # Look for patterns in each user of the Conv node
                    for user in node.users:
                        if user.op == 'call_module':
                            user_target = _get_module_qualname(gm, user.target)
                            
                            if user_target in modules and isinstance(modules[user_target], nn.BatchNorm2d):
                                # Found Conv-BN pair
                                bn_node = user
                                
                                # Look for ReLU after BN
                                for relu_candidate in bn_node.users:
                                    if relu_candidate.op == 'call_module':
                                        relu_target = _get_module_qualname(gm, relu_candidate.target)
                                        
                                        if relu_target in modules and isinstance(modules[relu_target], nn.ReLU):
                                            patterns.append([node, bn_node, relu_candidate])
                                            break
                                    elif relu_candidate.op == 'call_function' and relu_candidate.target == torch.nn.functional.relu:
                                        patterns.append([node, bn_node, relu_candidate])
                                        break
        
        return patterns
    
    def _trace_conv_bn_relu(self, conv_node: Node, gm: GraphModule, modules: Dict[str, nn.Module]) -> Optional[List[Node]]:
        """Trace Conv -> BN -> ReLU pattern."""
        # Check if Conv has exactly one user
        users = list(conv_node.users)
        if len(users) != 1:
            logger.debug(f"Skipping fusion: Conv node has {len(users)} users")
            return None
        
        bn_node = users[0]
        if bn_node.op != 'call_module':
            return None
        
        # Check if the node is a BatchNorm
        bn_target = _get_module_qualname(gm, bn_node.target)
        if bn_target not in modules or not isinstance(modules[bn_target], nn.BatchNorm2d):
            return None
        
        # Check if BN has exactly one user
        users = list(bn_node.users)
        if len(users) != 1:
            logger.debug(f"Skipping fusion: BN node has {len(users)} users")
            return None
        
        relu_node = users[0]
        
        # Check if it's a ReLU module or F.relu function
        if relu_node.op == 'call_module':
            relu_target = _get_module_qualname(gm, relu_node.target)
            if relu_target not in modules or not isinstance(modules[relu_target], nn.ReLU):
                return None
        elif relu_node.op == 'call_function':
            if relu_node.target != torch.nn.functional.relu:
                return None
        else:
            return None
        
        return [conv_node, bn_node, relu_node]
    
    def _trace_conv_relu(self, conv_node: Node, gm: GraphModule, modules: Dict[str, nn.Module]) -> Optional[List[Node]]:
        """Trace Conv -> ReLU pattern."""
        # Check if Conv has exactly one user
        users = list(conv_node.users)
        if len(users) != 1:
            logger.debug(f"Skipping fusion: Conv node has {len(users)} users")
            return None
        
        relu_node = users[0]
        
        # Check if it's a ReLU module or F.relu function
        if relu_node.op == 'call_module':
            relu_target = _get_module_qualname(gm, relu_node.target)
            if relu_target not in modules or not isinstance(modules[relu_target], nn.ReLU):
                return None
        elif relu_node.op == 'call_function':
            if relu_node.target != torch.nn.functional.relu:
                return None
        else:
            return None
        
        return [conv_node, relu_node]
    
    def _trace_linear_relu(self, linear_node: Node, gm: GraphModule, modules: Dict[str, nn.Module]) -> Optional[List[Node]]:
        """Trace Linear -> ReLU pattern."""
        # Check if Linear has exactly one user
        users = list(linear_node.users)
        if len(users) != 1:
            logger.debug(f"Skipping fusion: Linear node has {len(users)} users")
            return None
        
        relu_node = users[0]
        
        # Check if it's a ReLU module or F.relu function
        if relu_node.op == 'call_module':
            relu_target = _get_module_qualname(gm, relu_node.target)
            if relu_target not in modules or not isinstance(modules[relu_target], nn.ReLU):
                return None
        elif relu_node.op == 'call_function':
            if relu_node.target != torch.nn.functional.relu:
                return None
        else:
            return None
        
        return [linear_node, relu_node]

class ConvBatchNormReLUFusion(FusionPass):
    """Fusion pass for Conv2d-BatchNorm2d-ReLU patterns."""
    
    def __init__(self):
        super().__init__("Conv2d-BatchNorm2d-ReLU Fusion")
    
    def _trace_conv_bn_relu(self, conv_node: Node, gm: GraphModule, modules: Dict[str, nn.Module]) -> Optional[List[Node]]:
        """Trace Conv -> BN -> ReLU pattern starting from a Conv node."""
        # Get the single user of the Conv node
        users = list(conv_node.users)
        if len(users) != 1:
            return None
        
        bn_node = users[0]
        if bn_node.op != 'call_module':
            return None
        
        # Check if it's a BatchNorm2d
        bn_target = _get_module_qualname(gm, bn_node.target)
        if bn_target not in modules or not isinstance(modules[bn_target], nn.BatchNorm2d):
            return None
        
        # Get the single user of the BN node
        bn_users = list(bn_node.users)
        if len(bn_users) != 1:
            return None
        
        relu_node = bn_users[0]
        if relu_node.op != 'call_function' or relu_node.target != torch.nn.functional.relu:
            return None
        
        return [conv_node, bn_node, relu_node]

def apply_fusion_passes(model: torch.nn.Module) -> Dict[str, FusionResult]:
    """Apply all fusion passes to a model."""
    # Make a copy of the model to avoid modifying the original
    model = model.train(False)  # Set to eval mode for consistent fusion
    
    # Try to convert the model to a GraphModule if it's not already
    if not isinstance(model, GraphModule):
        try:
            # Use symbolic_trace to convert the model to a GraphModule
            model = fx.symbolic_trace(model)
        except Exception as e:
            logger.warning(f"Failed to symbolically trace model: {e}")
            # If tracing fails, we can't apply fusion passes
            empty_result = FusionResult()
            return {"ConvBNReLU": empty_result}
    
    results = {}
    
    # Apply fusion passes
    conv_bn_relu_fusion = ConvBatchNormReLUFusion()
    results["ConvBNReLU"] = conv_bn_relu_fusion.apply(model)
    
    return results


def print_fusion_results(fusion_results: Dict[str, FusionResult]):
    """Print the results of fusion passes."""
    for name, result in fusion_results.items():
        print(f"{name}  ")
        print(result)
        
        if result.fusion_details and result.total_fused_pairs > 0:
            print("\nFusion Details:")
            for module_path, pattern_name in result.fusion_details:
                print(f"  - {pattern_name} at {module_path}")
        
        print()


def optimize_model(model: torch.nn.Module) -> Tuple[torch.nn.Module, Dict[str, FusionResult]]:
    """Apply all optimization passes to a model."""
    # Apply fusion passes
    fusion_results = apply_fusion_passes(model)
    
    # Return the optimized model and fusion results
    return model, fusion_results