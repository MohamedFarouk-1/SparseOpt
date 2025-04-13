"""
Base classes for graph optimization passes.
"""

import torch
import torch.fx as fx
from typing import Dict, Any, Tuple, List, Optional, Set
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
    
    def _get_node_inputs(self, node: fx.Node) -> List[fx.Node]:
        """
        Get all input nodes for a given node.
        
        Args:
            node: The node to get inputs for
            
        Returns:
            List of input nodes
        """
        return [arg for arg in node.args if isinstance(arg, fx.Node)]
    
    def _get_node_users(self, node: fx.Node, graph: fx.Graph) -> List[fx.Node]:
        """
        Get all nodes that use the output of a given node.
        
        Args:
            node: The node to get users for
            graph: The graph containing the node
            
        Returns:
            List of user nodes
        """
        users = []
        for n in graph.nodes:
            if node in self._get_node_inputs(n):
                users.append(n)
        return users
    
    def _is_node_used(self, node: fx.Node, graph: fx.Graph) -> bool:
        """
        Check if a node's output is used by any other node.
        
        Args:
            node: The node to check
            graph: The graph containing the node
            
        Returns:
            True if the node's output is used, False otherwise
        """
        return len(self._get_node_users(node, graph)) > 0
    
    def _is_constant(self, node: fx.Node) -> bool:
        """
        Check if a node represents a constant value.
        
        Args:
            node: The node to check
            
        Returns:
            True if the node represents a constant, False otherwise
        """
        return (
            node.op == "get_attr" or
            (node.op == "call_function" and node.target == torch.tensor) or
            (isinstance(node.target, torch.Tensor) and node.target.requires_grad is False)
        )
    
    def _get_constant_value(self, node: fx.Node) -> Optional[torch.Tensor]:
        """
        Get the constant value represented by a node.
        
        Args:
            node: The node to get the constant value for
            
        Returns:
            The constant value, or None if the node does not represent a constant
        """
        if node.op == "get_attr":
            # For get_attr nodes, we need to access the actual tensor
            # This is a bit tricky since we don't have direct access to the module
            # For now, we'll just return None
            return None
        elif node.op == "call_function" and node.target == torch.tensor:
            # For torch.tensor calls, we can get the value from the args
            if len(node.args) == 1 and isinstance(node.args[0], (int, float, list, torch.Tensor)):
                return torch.tensor(node.args[0])
        elif isinstance(node.target, torch.Tensor) and node.target.requires_grad is False:
            # For direct tensor references
            return node.target
        
        return None
    
    def _is_identity_op(self, node: fx.Node) -> bool:
        """
        Check if a node represents an identity operation.
        
        Args:
            node: The node to check
            
        Returns:
            True if the node represents an identity operation, False otherwise
        """
        if node.op != "call_function":
            return False
        
        # Check for common identity operations
        if node.target == torch.nn.functional.relu:
            # ReLU is identity for non-negative inputs
            return False  # We can't determine this statically
        
        if node.target == torch.nn.functional.dropout:
            # Dropout with p=0 is identity
            if len(node.args) > 1 and node.args[1] == 0:
                return True
        
        if node.target == torch.nn.functional.layer_norm:
            # LayerNorm with default parameters is close to identity
            # We'll be conservative and not consider it identity
            return False
        
        # Check for casting to the same type
        if node.target == torch.Tensor.to:
            if len(node.args) > 1 and node.args[1] == node.args[0].dtype:
                return True
        
        return False
    
    def _is_activation(self, node: fx.Node) -> bool:
        """
        Check if a node represents an activation function.
        
        Args:
            node: The node to check
            
        Returns:
            True if the node represents an activation function, False otherwise
        """
        if node.op != "call_function":
            return False
        
        return node.target in [
            torch.nn.functional.relu,
            torch.nn.functional.gelu,
            torch.nn.functional.sigmoid,
            torch.nn.functional.tanh,
            torch.nn.functional.softmax,
            torch.nn.functional.log_softmax
        ]
    
    def _is_linear(self, node: fx.Node) -> bool:
        """
        Check if a node represents a linear operation.
        
        Args:
            node: The node to check
            
        Returns:
            True if the node represents a linear operation, False otherwise
        """
        if node.op != "call_module":
            return False
        
        return isinstance(node.target, str) and "linear" in node.target.lower()
    
    def _is_conv(self, node: fx.Node) -> bool:
        """
        Check if a node represents a convolution operation.
        
        Args:
            node: The node to check
            
        Returns:
            True if the node represents a convolution operation, False otherwise
        """
        if node.op != "call_module":
            return False
        
        return isinstance(node.target, str) and "conv" in node.target.lower()
    
    def _is_batch_norm(self, node: fx.Node) -> bool:
        """
        Check if a node represents a batch normalization operation.
        
        Args:
            node: The node to check
            
        Returns:
            True if the node represents a batch normalization operation, False otherwise
        """
        if node.op != "call_module":
            return False
        
        return isinstance(node.target, str) and "bn" in node.target.lower() or "batchnorm" in node.target.lower()
    
    def _is_dropout(self, node: fx.Node) -> bool:
        """
        Check if a node represents a dropout operation.
        
        Args:
            node: The node to check
            
        Returns:
            True if the node represents a dropout operation, False otherwise
        """
        if node.op == "call_function":
            return node.target == torch.nn.functional.dropout
        
        if node.op == "call_module":
            return isinstance(node.target, str) and "dropout" in node.target.lower()
        
        return False
    
    def _is_lightweight_op(self, node: fx.Node) -> bool:
        """
        Check if a node represents a lightweight operation.
        
        Args:
            node: The node to check
            
        Returns:
            True if the node represents a lightweight operation, False otherwise
        """
        return (
            self._is_activation(node) or
            self._is_dropout(node) or
            (node.op == "call_function" and node.target == torch.add) or
            (node.op == "call_function" and node.target == torch.mul)
        )
    
    def _is_compute_intensive_op(self, node: fx.Node) -> bool:
        """
        Check if a node represents a compute-intensive operation.
        
        Args:
            node: The node to check
            
        Returns:
            True if the node represents a compute-intensive operation, False otherwise
        """
        return (
            self._is_linear(node) or
            self._is_conv(node) or
            self._is_batch_norm(node) or
            (node.op == "call_function" and node.target == torch.matmul) or
            (node.op == "call_function" and node.target == torch.bmm)
        )
    
    def _get_node_metadata(self, node: fx.Node) -> Dict[str, Any]:
        """
        Get metadata about a node.
        
        Args:
            node: The node to get metadata for
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            "op": node.op,
            "target": str(node.target),
            "args": [str(arg) for arg in node.args],
            "kwargs": {k: str(v) for k, v in node.kwargs.items()},
            "is_constant": self._is_constant(node),
            "is_identity": self._is_identity_op(node),
            "is_activation": self._is_activation(node),
            "is_linear": self._is_linear(node),
            "is_conv": self._is_conv(node),
            "is_batch_norm": self._is_batch_norm(node),
            "is_dropout": self._is_dropout(node),
            "is_lightweight": self._is_lightweight_op(node),
            "is_compute_intensive": self._is_compute_intensive_op(node)
        }
        
        return metadata 