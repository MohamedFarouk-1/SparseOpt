"""
Example: optimize ResNet-50 with SparseOpt.

Demonstrates Conv+BN+ReLU fusion, dead-node elimination, and node reordering
on a standard CNN via the Python API.
"""

import torch
from sparseopt import optimize_model, get_demo_model, measure_latency_and_memory
from rich.console import Console

console = Console()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"\n[bold]ResNet-50 optimization demo[/bold]  (device: {device})\n")

    # Load model
    model, inputs = get_demo_model("resnet50", device=device)
    console.print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Baseline
    before = measure_latency_and_memory(model, inputs, device=device)
    console.print(f"Baseline latency:  {before['latency_ms']:.2f} ms")

    # Optimize — only Conv+BN+ReLU fusion + structural passes
    optimized, stats = optimize_model(
        model,
        inputs,
        passes=["dropout", "dead_node", "reorder", "conv_fusion"],
        verify=True,
    )
    console.print(f"\nOptimization stats: {stats}")

    # Optimized benchmark
    after = measure_latency_and_memory(optimized, inputs, device=device)
    console.print(f"Optimized latency: {after['latency_ms']:.2f} ms")

    speedup = before["latency_ms"] / after["latency_ms"]
    console.print(f"Speedup: [green]{speedup:.2f}×[/green]")


if __name__ == "__main__":
    main()
