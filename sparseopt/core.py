"""
Core optimization entry point for SparseOpt.

Provides optimize_model() — a single function that accepts any nn.Module,
traces it with torch.fx, applies the configured optimization passes, verifies
numerical correctness, and returns the optimized module with statistics.

For models that cannot be whole-graph traced (e.g. BERT, GPT-2), it
automatically falls back to layer-by-layer optimization.
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.fx as fx
import torch.nn as nn

from .graph.passes.dead_node import DeadNodeEliminationPass
from .graph.passes.dropout_elimination import DropoutElimination
from .graph.passes.reordering import NodeReorderingPass
from .graph.passes.linear_fusion import LinearGELUFusion, LinearReLUFusion
from .graph.passes.linear_layernorm_fusion import LinearLayerNormFusion
from .graph.passes.conv_fusion import ConvBatchNormReLUFusion
from .graph.optimizer_pipeline import OptimizerPipeline

logger = logging.getLogger(__name__)

# Registry maps CLI/API pass names → pass classes
PASS_REGISTRY = {
    "dropout":          DropoutElimination,
    "dead_node":        DeadNodeEliminationPass,
    "reorder":          NodeReorderingPass,
    "conv_fusion":      ConvBatchNormReLUFusion,
    "linear_gelu":      LinearGELUFusion,
    "linear_relu":      LinearReLUFusion,
    "linear_layernorm": LinearLayerNormFusion,
}

# Default pass order for whole-graph optimization
DEFAULT_PASSES = [
    "dropout",
    "dead_node",
    "reorder",
    "conv_fusion",
    "linear_relu",
    "linear_gelu",
    "linear_layernorm",
]

# Passes that are safe to apply per-submodule (no structural graph changes)
LAYER_PASSES = ["linear_relu", "linear_gelu", "linear_layernorm", "dropout"]


def optimize_model(
    model: nn.Module,
    example_inputs: Dict[str, torch.Tensor],
    passes: Optional[List[str]] = None,
    verify: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Optimize a PyTorch model using torch.fx graph passes.

    Tries whole-graph symbolic tracing first. If tracing fails (e.g. BERT,
    GPT-2 with dynamic control flow), falls back to layer-by-layer optimization
    where each traceable submodule is optimized independently.

    Args:
        model:          The nn.Module to optimize. Will be deep-copied.
        example_inputs: Dict mapping argument names to tensors matching the
                        model's forward() signature.
        passes:         List of pass names to apply (default: all passes).
                        Valid names: dropout, dead_node, reorder, conv_fusion,
                        linear_relu, linear_gelu, linear_layernorm.
        verify:         Whether to verify numerical correctness after each pass
                        (whole-graph path only). Uses torch.allclose(rtol=1e-4).

    Returns:
        (optimized_model, stats) where stats contains:
            method          — "full_graph" or "layer_by_layer"
            node_count_before / node_count_after  (full_graph only)
            total_fusions
            correct         — bool (full_graph + verify only)
            per-pass stats
    """
    if passes is None:
        passes = DEFAULT_PASSES

    model = copy.deepcopy(model).eval()

    # Attempt whole-graph tracing
    try:
        gm = fx.symbolic_trace(model)
        node_count_before = len(list(gm.graph.nodes))

        pipeline = _build_pipeline(passes, verify_correctness=verify, benchmark=False)
        gm, pipe_stats = pipeline.optimize(gm, example_inputs)
        gm.recompile()

        node_count_after = len(list(gm.graph.nodes))
        total_fusions = _count_fusions(pipe_stats)

        stats: Dict[str, Any] = {
            "method": "full_graph",
            "node_count_before": node_count_before,
            "node_count_after": node_count_after,
            "nodes_eliminated": node_count_before - node_count_after,
            "total_fusions": total_fusions,
        }
        # Propagate correctness check
        correct_key = next(
            (k for k in pipe_stats if k.endswith("_correct")), None
        )
        if correct_key is not None:
            stats["correct"] = pipe_stats[correct_key]
        stats.update(pipe_stats)
        return gm, stats

    except Exception as trace_err:
        logger.debug("Full-graph trace failed (%s), falling back to layer-by-layer.", trace_err)

    # Layer-by-layer fallback
    stats = _optimize_layer_by_layer(model, passes)
    return model, stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_pipeline(
    pass_names: List[str],
    verify_correctness: bool = True,
    benchmark: bool = False,
) -> OptimizerPipeline:
    """Instantiate an OptimizerPipeline from a list of pass names."""
    pass_instances = []
    for name in pass_names:
        cls = PASS_REGISTRY.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown pass '{name}'. Valid options: {list(PASS_REGISTRY)}"
            )
        pass_instances.append(cls())
    return OptimizerPipeline(
        passes=pass_instances,
        verify_correctness=verify_correctness,
        benchmark=benchmark,
    )


def _optimize_layer_by_layer(
    model: nn.Module, pass_names: List[str]
) -> Dict[str, Any]:
    """Apply fusion passes to each individually-traceable submodule."""
    safe_pass_names = [p for p in pass_names if p in LAYER_PASSES]
    stats: Dict[str, Any] = {
        "method": "layer_by_layer",
        "layers_traced": 0,
        "layers_fused": 0,
        "total_fusions": 0,
    }

    if not safe_pass_names:
        return stats

    pass_instances = [PASS_REGISTRY[p]() for p in safe_pass_names]

    for name, module in list(model.named_modules()):
        if not name:  # skip root
            continue
        # Only attempt compound modules that contain at least one Linear
        if not any(isinstance(m, nn.Linear) for m in module.modules()):
            continue
        # Skip plain leaf layers
        if not list(module.children()):
            continue

        try:
            gm = fx.symbolic_trace(module)
        except Exception:
            continue

        stats["layers_traced"] += 1
        fusions_this_layer = 0

        for pass_instance in pass_instances:
            try:
                gm, pass_stats = pass_instance.apply(gm, {})
                fusions_this_layer += pass_stats.get("fused_patterns", 0)
                fusions_this_layer += pass_stats.get("eliminated_dropouts", 0)
            except Exception:
                continue

        if fusions_this_layer > 0:
            try:
                gm.recompile()
                _set_submodule(model, name, gm)
                stats["layers_fused"] += 1
                stats["total_fusions"] += fusions_this_layer
            except Exception:
                pass

    return stats


def _set_submodule(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace a named submodule within the model hierarchy."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _count_fusions(stats: Dict[str, Any]) -> int:
    """Sum all fusion counts across all passes in a pipeline stats dict."""
    total = 0
    for v in stats.values():
        if isinstance(v, int):
            total += v
    return total
