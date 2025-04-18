#!/usr/bin/env python
"""
SparseOpt Colab Example

This script demonstrates how to use SparseOpt to optimize and benchmark HuggingFace models in Google Colab.
"""

import torch
import time
from rich.console import Console
from rich.table import Table

# Import SparseOpt modules
from sparseopt.huggingface import (
    optimize_hf_model, 
    load_hf_model, 
    create_hf_input,
    benchmark_model,
    print_benchmark_results
)

console = Console()

def validate_numerical_correctness(
    original_model, 
    optimized_model, 
    inputs, 
    tolerance=1e-5
):
    """Validate that the optimized model produces the same outputs as the original model."""
    console.print("[bold blue]Validating numerical correctness...[/bold blue]")
    
    # Run inference on both models
    with torch.no_grad():
        original_output = original_model(**inputs)
        optimized_output = optimized_model(**inputs)
    
    # Handle different output types
    if isinstance(original_output, tuple):
        original_output = original_output[0]
    if isinstance(optimized_output, tuple):
        optimized_output = optimized_output[0]
    
    # Compare outputs
    is_close = torch.allclose(original_output, optimized_output, rtol=tolerance, atol=tolerance)
    
    if is_close:
        console.print("[bold green]✓ Numerical correctness validated![/bold green]")
    else:
        console.print("[bold red]✗ Numerical correctness check failed![/bold red]")
        console.print("[yellow]Outputs differ beyond the specified tolerance.[/yellow]")
    
    return is_close

def run_benchmark(
    model_name,
    device="cuda",
    num_runs=10,
    warmup=3,
    text="Hello, this is SparseOpt!",
    max_length=128
):
    """Run benchmark on a HuggingFace model."""
    console.print(f"[bold blue]Benchmarking {model_name} on {device}...[/bold blue]")
    
    # Load model and tokenizer
    model, tokenizer = load_hf_model(model_name, device=device)
    
    # Create input
    inputs = create_hf_input(tokenizer, text, device=device, max_length=max_length)
    
    # Benchmark original model
    console.print("[bold blue]Benchmarking original model...[/bold blue]")
    original_results = benchmark_model(model, inputs, device=device, num_runs=num_runs, warmup=warmup)
    
    # Optimize model
    console.print("[bold blue]Optimizing model...[/bold blue]")
    optimized_model, fusion_stats = optimize_hf_model(model_name, device=device)
    
    # Benchmark optimized model
    console.print("[bold blue]Benchmarking optimized model...[/bold blue]")
    optimized_results = benchmark_model(optimized_model, inputs, device=device, num_runs=num_runs, warmup=warmup)
    
    # Validate numerical correctness
    validate_numerical_correctness(model, optimized_model, inputs)
    
    # Print benchmark results
    print_benchmark_results(original_results, optimized_results, fusion_stats)
    
    # Print fusion details
    console.print("\n[bold blue]Fusion Details:[/bold blue]")
    fusion_table = Table()
    fusion_table.add_column("Fusion Type", style="cyan")
    fusion_table.add_column("Count", style="green")
    
    for fusion_type, fusions in fusion_stats.get("fusion_results", {}).items():
        fusion_table.add_row(fusion_type, str(len(fusions)))
    
    console.print(fusion_table)
    
    # Print summary
    speedup = original_results["mean_latency"] / optimized_results["mean_latency"]
    console.print(f"\n[bold green]Summary:[/bold green]")
    console.print(f"Model: {model_name}")
    console.print(f"Baseline ({device}): {original_results['mean_latency']:.2f}ms | Optimized: {optimized_results['mean_latency']:.2f}ms | Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold blue]Using device: {device}[/bold blue]")
    
    # Run benchmark on a few models
    models = [
        "bert-base-uncased",
        "distilbert-base-uncased",
        "facebook/opt-350m"
    ]
    
    for model_name in models:
        try:
            run_benchmark(model_name, device=device)
            console.print("\n" + "="*50 + "\n")
        except Exception as e:
            console.print(f"[bold red]Error benchmarking {model_name}:[/bold red] {str(e)}")
            console.print("\n" + "="*50 + "\n") 