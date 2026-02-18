"""
Example: optimize BERT with SparseOpt (layer-by-layer).

BERT's dynamic attention masking prevents whole-graph torch.fx tracing.
SparseOpt automatically falls back to layer-by-layer optimization, applying
Linear+GELU and Linear+LayerNorm fusions to each traceable submodule.
"""

import torch
from sparseopt import optimize_model, get_demo_model, measure_latency_and_memory
from rich.console import Console

console = Console()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"\n[bold]BERT-base optimization demo[/bold]  (device: {device})\n")
    console.print("[dim]Note: BERT uses layer-by-layer optimization (dynamic control flow)[/dim]\n")

    # Load model
    model, inputs = get_demo_model("bert-base", device=device)
    console.print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    # Baseline
    before = measure_latency_and_memory(model, inputs, device=device, num_runs=20, warmup=5)
    console.print(f"Baseline latency:  {before['latency_ms']:.2f} ms")

    # Optimize — layer-by-layer with fusion passes
    optimized, stats = optimize_model(
        model,
        inputs,
        passes=["dropout", "linear_gelu", "linear_relu", "linear_layernorm"],
        verify=False,   # layer-by-layer path skips whole-model verification
    )
    console.print(f"\nOptimization stats:")
    console.print(f"  Strategy:         {stats.get('method')}")
    console.print(f"  Layers traced:    {stats.get('layers_traced', 0)}")
    console.print(f"  Layers fused:     {stats.get('layers_fused', 0)}")
    console.print(f"  Total fusions:    {stats.get('total_fusions', 0)}")

    # Optimized benchmark
    after = measure_latency_and_memory(optimized, inputs, device=device, num_runs=20, warmup=5)
    console.print(f"\nOptimized latency: {after['latency_ms']:.2f} ms")

    speedup = before["latency_ms"] / after["latency_ms"]
    console.print(f"Speedup: [green]{speedup:.2f}×[/green]")


if __name__ == "__main__":
    main()
