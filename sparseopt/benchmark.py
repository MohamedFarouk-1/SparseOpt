"""
Benchmarking utilities for SparseOpt.

Provides measure_latency_and_memory() — the single function used by the CLI
and examples to characterise a model before and after optimization.
"""

import time
from typing import Any, Dict

import torch
import torch.nn as nn


def measure_latency_and_memory(
    model: nn.Module,
    inputs: Dict[str, Any],
    device: str = "cpu",
    num_runs: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """Measure inference latency and peak memory usage.

    Runs `warmup` forward passes (discarded), then `num_runs` timed passes.

    Args:
        model:    The model to benchmark (must already be on `device`).
        inputs:   Dict of tensors matching the model's forward() signature.
        device:   "cpu" or "cuda".
        num_runs: Number of timed iterations.
        warmup:   Number of warmup iterations before timing begins.

    Returns:
        {
            "latency_ms":     mean latency over num_runs (milliseconds),
            "latency_std_ms": std dev of latency (milliseconds),
            "peak_memory_mb": peak memory during timed run (MB),
        }
    """
    model.eval()

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _forward(model, inputs)

        # Memory baseline and reset
        if device == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            mem_before = torch.cuda.memory_allocated(device)
        else:
            mem_before = _cpu_memory_mb()

        # Timed runs
        latencies = []
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _forward(model, inputs)
            if device == "cuda":
                torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        # Peak memory
        if device == "cuda":
            peak_bytes = torch.cuda.max_memory_allocated(device)
            peak_mb = (peak_bytes - mem_before) / (1024 ** 2)
        else:
            mem_after = _cpu_memory_mb()
            peak_mb = max(mem_after - mem_before, 0.0)

    n = len(latencies)
    mean_lat = sum(latencies) / n
    variance = sum((x - mean_lat) ** 2 for x in latencies) / n
    std_lat = variance ** 0.5

    return {
        "latency_ms": mean_lat,
        "latency_std_ms": std_lat,
        "peak_memory_mb": peak_mb,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _forward(model: nn.Module, inputs: Dict[str, Any]) -> Any:
    with torch.no_grad():
        return model(**inputs)


def _cpu_memory_mb() -> float:
    """Return current process RSS in MB using psutil (if available)."""
    try:
        import psutil
        import os
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except ImportError:
        return 0.0
