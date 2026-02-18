#!/usr/bin/env python3
"""
Run all SparseOpt benchmarks and write results to benchmarks/results.json.

Usage:
    python benchmarks/run_benchmarks.py
    python benchmarks/run_benchmarks.py --device cuda
    python benchmarks/run_benchmarks.py --runs 50 --warmup 5
"""

import argparse
import json
import os
import sys
import time

# Allow running from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sparseopt import optimize_model, get_demo_model, measure_latency_and_memory

MODELS = ["resnet50", "mlp", "bert-base"]
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")


def run_one(model_name: str, device: str, num_runs: int, warmup: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  {model_name}  ({device})")
    print(f"{'='*60}")

    model, inputs = get_demo_model(model_name, device=device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params/1e6:.1f}M")

    # Baseline
    print("  Benchmarking baseline...", end=" ", flush=True)
    before = measure_latency_and_memory(model, inputs, device=device,
                                        num_runs=num_runs, warmup=warmup)
    print(f"{before['latency_ms']:.2f} ms")

    # Optimize
    print("  Optimizing...", end=" ", flush=True)
    t0 = time.perf_counter()
    optimized, stats = optimize_model(model, inputs, verify=(model_name != "bert-base"))
    opt_ms = (time.perf_counter() - t0) * 1000
    print(f"done in {opt_ms:.0f} ms  "
          f"(method={stats.get('method','?')}, "
          f"fusions={stats.get('total_fusions',0)})")

    # Optimized
    print("  Benchmarking optimized...", end=" ", flush=True)
    after = measure_latency_and_memory(optimized, inputs, device=device,
                                       num_runs=num_runs, warmup=warmup)
    print(f"{after['latency_ms']:.2f} ms")

    speedup = before["latency_ms"] / after["latency_ms"]
    pct = (after["latency_ms"] - before["latency_ms"]) / before["latency_ms"] * 100
    print(f"  Speedup: {speedup:.2f}×  ({pct:+.1f}%)")

    return {
        "model": model_name,
        "device": device,
        "parameters_M": round(params / 1e6, 2),
        "before": before,
        "after": after,
        "speedup": round(speedup, 3),
        "latency_reduction_pct": round(pct, 2),
        "optimization_time_ms": round(opt_ms, 1),
        "stats": {
            k: v for k, v in stats.items()
            if isinstance(v, (int, float, bool, str))
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--runs",   type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    all_results = []
    for name in MODELS:
        try:
            result = run_one(name, args.device, args.runs, args.warmup)
            all_results.append(result)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
