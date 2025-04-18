#!/usr/bin/env python
"""
SparseOpt CLI for optimizing and benchmarking HuggingFace models.

Usage:
    python optimize.py --model bert-base-uncased --device cuda
"""

import argparse
import os
import time
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Union
from rich.console import Console
from rich.table import Table

from sparseopt.huggingface import (
    optimize_hf_model, 
    load_hf_model, 
    create_hf_input,
    benchmark_model,
    print_benchmark_results
)
from sparseopt.transformer_fusion import apply_transformer_fusion_passes

console = Console()

def validate_numerical_correctness(
    original_model: nn.Module,
    optimized_model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    tolerance: float = 1e-5
) -> bool:
    """
    Validate that the optimized model produces the same outputs as the original model.
    
    Args:
        original_model: The original model
        optimized_model: The optimized model
        inputs: Input tensors
        tolerance: Tolerance for numerical differences
        
    Returns:
        True if outputs match within tolerance, False otherwise
    """
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
    model_name: str,
    device: str = "cuda",
    num_runs: int = 10,
    warmup: int = 3,
    text: str = "Hello, this is SparseOpt!",
    max_length: int = 128
) -> None:
    """
    Run benchmark on a HuggingFace model.
    
    Args:
        model_name: Name of the HuggingFace model
        device: Device to run on
        num_runs: Number of benchmark runs
        warmup: Number of warmup runs
        text: Input text for benchmarking
        max_length: Maximum sequence length
    """
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

def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="SparseOpt CLI for optimizing HuggingFace models")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda", help="Device to run on")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup runs")
    parser.add_argument("--text", type=str, default="Hello, this is SparseOpt!", help="Input text for benchmarking")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Check if CUDA is available if device is set to cuda
    if args.device == "cuda" and not torch.cuda.is_available():
        console.print("[bold red]Error:[/bold red] CUDA is not available. Please use --device cpu")
        return
    
    try:
        run_benchmark(
            args.model,
            device=args.device,
            num_runs=args.num_runs,
            warmup=args.warmup,
            text=args.text,
            max_length=args.max_length
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

if __name__ == "__main__":
    main() 