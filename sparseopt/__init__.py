"""
SparseOpt — PyTorch FX computation graph optimizer.

Quick start:
    from sparseopt import optimize_model, get_demo_model

    model, inputs = get_demo_model("resnet50", device="cpu")
    optimized, stats = optimize_model(model, inputs)
    print(stats)
"""

from .core import optimize_model, PASS_REGISTRY, DEFAULT_PASSES
from .models import get_demo_model
from .benchmark import measure_latency_and_memory

__version__ = "0.1.0"
__all__ = [
    "optimize_model",
    "get_demo_model",
    "measure_latency_and_memory",
    "PASS_REGISTRY",
    "DEFAULT_PASSES",
]
