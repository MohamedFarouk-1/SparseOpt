"""
Example: optimize a custom MLP with SparseOpt.

Demonstrates Linear+ReLU fusion and Dropout elimination on a simple
fully-connected network via the Python API.
"""

import torch
from sparseopt import optimize_model, get_demo_model, measure_latency_and_memory
from rich.console import Console

console = Console()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"\n[bold]MLP optimization demo[/bold]  (device: {device})\n")

    # Load model
    model, inputs = get_demo_model("mlp", device=device)
    console.print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Baseline
    before = measure_latency_and_memory(model, inputs, device=device)
    console.print(f"Baseline latency:  {before['latency_ms']:.3f} ms")

    # Optimize — Dropout elimination + Linear+ReLU fusion
    optimized, stats = optimize_model(
        model,
        inputs,
        passes=["dropout", "dead_node", "reorder", "linear_relu", "linear_gelu"],
        verify=True,
    )
    console.print(f"\nOptimization stats: {stats}")

    # Optimized benchmark
    after = measure_latency_and_memory(optimized, inputs, device=device)
    console.print(f"Optimized latency: {after['latency_ms']:.3f} ms")

    speedup = before["latency_ms"] / after["latency_ms"]
    console.print(f"Speedup: [green]{speedup:.2f}×[/green]")

    correct = stats.get("correct", None)
    if correct is not None:
        label = "[green]✓ pass[/green]" if correct else "[red]✗ fail[/red]"
        console.print(f"Numerical correctness (tol=1e-4): {label}")


if __name__ == "__main__":
    main()
