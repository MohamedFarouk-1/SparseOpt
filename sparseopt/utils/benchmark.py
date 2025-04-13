"""
Benchmarking utilities for SparseOpt.

This module provides utilities for benchmarking model performance.
"""

import torch
import time
from typing import Dict, Any, Callable, Optional, Union, List
from rich.console import Console
from rich.table import Table

console = Console()

def benchmark_model(
    model: torch.nn.Module,
    inputs: Union[Dict[str, torch.Tensor], Tuple[Any, ...]],
    device: Optional[str] = None,
    num_runs: int = 100,
    warmup: int = 10,
    batch_size: int = 1,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Benchmark model performance with GPU support.
    
    Args:
        model: Model to benchmark
        inputs: Dictionary of input tensors or tuple of inputs
        device: Device to run on (auto-detected if None)
        num_runs: Number of benchmark runs
        warmup: Number of warmup runs
        batch_size: Batch size for inputs
        verbose: Whether to print benchmark results
        
    Returns:
        Dictionary of benchmark results
    """
    # Auto-detect device if not provided
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Prepare inputs
    if isinstance(inputs, dict):
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        inputs = tuple(x.to(device) if torch.is_tensor(x) else x for x in inputs)
    
    # Warmup runs
    for _ in range(warmup):
        with torch.no_grad():
            if isinstance(inputs, dict):
                _ = model(**inputs)
            else:
                _ = model(*inputs)
    
    # Benchmark runs
    latencies = []
    for _ in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        
        start = time.time()
        with torch.no_grad():
            if isinstance(inputs, dict):
                _ = model(**inputs)
            else:
                _ = model(*inputs)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        end = time.time()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    # Calculate statistics
    latencies = torch.tensor(latencies)
    mean_latency = latencies.mean().item()
    std_latency = latencies.std().item()
    
    results = {
        "mean_latency_ms": mean_latency,
        "std_latency_ms": std_latency,
        "min_latency_ms": latencies.min().item(),
        "max_latency_ms": latencies.max().item(),
        "device": device,
        "batch_size": batch_size
    }
    
    if verbose:
        print_benchmark_results(results)
    
    return results

def print_benchmark_results(results: Dict[str, float]) -> None:
    """
    Print benchmark results in a formatted table.
    
    Args:
        results: Dictionary of benchmark results
    """
    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Device", results.get("device", "unknown"))
    table.add_row("Batch Size", str(results.get("batch_size", 1)))
    table.add_row("Mean Latency (ms)", f"{results['mean_latency_ms']:.2f}")
    table.add_row("Std Latency (ms)", f"{results['std_latency_ms']:.2f}")
    table.add_row("Min Latency (ms)", f"{results['min_latency_ms']:.2f}")
    table.add_row("Max Latency (ms)", f"{results['max_latency_ms']:.2f}")
    
    console.print(table)

def compare_models(
    original_model: torch.nn.Module,
    optimized_model: torch.nn.Module,
    inputs: Union[Dict[str, torch.Tensor], Tuple[Any, ...]],
    device: Optional[str] = None,
    num_runs: int = 100,
    warmup: int = 10,
    batch_size: int = 1
) -> Dict[str, Any]:
    """
    Compare performance between original and optimized models.
    
    Args:
        original_model: Original model to benchmark
        optimized_model: Optimized model to benchmark
        inputs: Dictionary of input tensors or tuple of inputs
        device: Device to run on (auto-detected if None)
        num_runs: Number of benchmark runs
        warmup: Number of warmup runs
        batch_size: Batch size for inputs
        
    Returns:
        Dictionary with comparison results
    """
    # Benchmark original model
    original_results = benchmark_model(
        original_model, 
        inputs, 
        device=device, 
        num_runs=num_runs, 
        warmup=warmup, 
        batch_size=batch_size,
        verbose=False
    )
    
    # Benchmark optimized model
    optimized_results = benchmark_model(
        optimized_model, 
        inputs, 
        device=device, 
        num_runs=num_runs, 
        warmup=warmup, 
        batch_size=batch_size,
        verbose=False
    )
    
    # Calculate speedup
    speedup = original_results["mean_latency_ms"] / optimized_results["mean_latency_ms"]
    
    # Print comparison
    table = Table(title="Model Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Original", style="green")
    table.add_column("Optimized", style="green")
    table.add_column("Speedup", style="yellow")
    
    table.add_row(
        "Mean Latency (ms)",
        f"{original_results['mean_latency_ms']:.2f}",
        f"{optimized_results['mean_latency_ms']:.2f}",
        f"{speedup:.2f}x"
    )
    
    table.add_row(
        "Std Latency (ms)",
        f"{original_results['std_latency_ms']:.2f}",
        f"{optimized_results['std_latency_ms']:.2f}",
        "-"
    )
    
    console.print(table)
    
    return {
        "original": original_results,
        "optimized": optimized_results,
        "speedup": speedup
    } 