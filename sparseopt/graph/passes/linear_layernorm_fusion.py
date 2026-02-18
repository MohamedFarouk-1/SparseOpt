"""
Linear + LayerNorm fusion pass.

Fuses nn.Linear → nn.LayerNorm sequences into a single module,
reducing Python dispatch overhead common in transformer architectures.
"""

import torch
import torch.nn as nn
import torch.fx as fx
from typing import Dict, Any, Tuple
from ..base import GraphPass


class LinearLayerNorm(nn.Module):
    """Fused Linear-LayerNorm module."""

    def __init__(self, linear: nn.Linear, layernorm: nn.LayerNorm):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.data.clone())
        self.bias = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        self.layernorm = layernorm
        self.to(linear.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.linear(x, self.weight, self.bias)
        return self.layernorm(x)


class LinearLayerNormFusion(GraphPass):
    """
    Pass to fuse Linear and LayerNorm operations.

    Identifies Linear → LayerNorm patterns in the FX graph and replaces them
    with a single LinearLayerNorm module. This is common in transformer FFN blocks.
    """

    def apply(
        self,
        graph_module: fx.GraphModule,
        example_inputs: Tuple[Any, ...],
    ) -> Tuple[fx.GraphModule, Dict[str, Any]]:
        new_graph = fx.Graph()
        processed_nodes: Dict[fx.Node, fx.Node] = {}
        stats = {"fused_patterns": 0}

        for node in graph_module.graph.nodes:
            if node in processed_nodes:
                continue

            if self._is_linear_layernorm_pattern(node, graph_module):
                linear_node = node
                ln_node = list(linear_node.users)[0]

                linear_module = graph_module.get_submodule(linear_node.target)
                ln_module = graph_module.get_submodule(ln_node.target)
                fused_module = LinearLayerNorm(linear_module, ln_module)

                fused_name = f"{linear_node.target}_ln_fused"
                graph_module.add_module(fused_name, fused_module)

                new_node = new_graph.call_module(
                    fused_name,
                    args=self._copy_args(linear_node.args, processed_nodes),
                    kwargs=linear_node.kwargs,
                )
                processed_nodes[linear_node] = new_node
                processed_nodes[ln_node] = new_node
                stats["fused_patterns"] += 1

            else:
                if node.op == "placeholder":
                    new_node = new_graph.placeholder(node.name, type_expr=node.type)
                elif node.op == "output":
                    new_node = new_graph.output(
                        self._copy_args(node.args, processed_nodes)[0]
                    )
                elif node.op == "call_module":
                    new_node = new_graph.call_module(
                        node.target,
                        args=self._copy_args(node.args, processed_nodes),
                        kwargs=node.kwargs,
                    )
                elif node.op == "call_function":
                    new_node = new_graph.call_function(
                        node.target,
                        args=self._copy_args(node.args, processed_nodes),
                        kwargs=node.kwargs,
                    )
                else:
                    new_node = new_graph.node_copy(node, lambda x: processed_nodes[x])
                processed_nodes[node] = new_node

        return fx.GraphModule(graph_module, new_graph), stats

    def _is_linear_layernorm_pattern(
        self, node: fx.Node, graph_module: fx.GraphModule
    ) -> bool:
        if not self._is_linear(node, graph_module):
            return False
        if len(node.users) != 1:
            return False
        user = list(node.users)[0]
        return self._is_layernorm(user, graph_module)

    def _is_layernorm(self, node: fx.Node, graph_module: fx.GraphModule) -> bool:
        if node.op == "call_module":
            try:
                return isinstance(graph_module.get_submodule(node.target), nn.LayerNorm)
            except AttributeError:
                return False
        return False

    def _is_linear(self, node: fx.Node, graph_module: fx.GraphModule) -> bool:
        if node.op == "call_module":
            try:
                return isinstance(graph_module.get_submodule(node.target), nn.Linear)
            except AttributeError:
                return False
        return False

    def _copy_args(
        self,
        args: Tuple[Any, ...],
        processed_nodes: Dict[fx.Node, fx.Node],
    ) -> Tuple[Any, ...]:
        return tuple(
            processed_nodes[a] if isinstance(a, fx.Node) else a for a in args
        )
