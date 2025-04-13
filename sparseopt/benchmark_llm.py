"""Benchmarking utilities for LLMs and MoE models."""

import torch
import torch.nn as nn
import time
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from rich.console import Console
from rich.table import Table
import gc
import psutil
import os

console = Console()

class LLMBenchmarker:
    """Benchmarker for LLMs and MoE models."""
    
    def __init__(self, model: nn.Module, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize the benchmarker with a model.
        
        Args:
            model: PyTorch model to benchmark
            device: Device to run the benchmark on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    def generate_dummy_input(self, input_shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        """Generate dummy input for the model.
        
        Args:
            input_shape: Shape of the input tensor. If None, a default shape is used.
            
        Returns:
            Dummy input tensor
        """
        # Default input shape for transformer models
        if input_shape is None:
            # Common shape for transformer models: [batch_size, sequence_length]
            input_shape = (1, 128)
            
        # Generate random input
        dummy_input = torch.randn(input_shape, device=self.device)
        
        return dummy_input
        
    def measure_latency(self, input_data: Optional[torch.Tensor] = None, 
                        warmup_runs: int = 5, 
                        timed_runs: int = 20) -> float:
        """Measure the latency of the model.
        
        Args:
            input_data: Input data to use for benchmarking. If None, dummy input is generated.
            warmup_runs: Number of warmup runs
            timed_runs: Number of timed runs
            
        Returns:
            Average latency in seconds
        """
        # Generate dummy input if none provided
        if input_data is None:
            input_data = self.generate_dummy_input()
            
        # Warmup runs
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = self.model(input_data)
                
        # Synchronize CUDA if using GPU
        if self.device == "cuda":
            torch.cuda.synchronize()
            
        # Timed runs
        latencies = []
        for _ in range(timed_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = self.model(input_data)
            if self.device == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            latencies.append(end_time - start_time)
            
        # Calculate average latency
        avg_latency = sum(latencies) / len(latencies)
        
        # Print results
        console.print(f"[green]✓[/green] Average latency: {avg_latency * 1000:.2f} ms")
        
        return avg_latency
        
    def measure_memory(self, input_data: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Measure the memory usage of the model.
        
        Args:
            input_data: Input data to use for benchmarking. If None, dummy input is generated.
            
        Returns:
            Dictionary with memory usage metrics
        """
        # Generate dummy input if none provided
        if input_data is None:
            input_data = self.generate_dummy_input()
            
        # Clear cache
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        # Measure memory before forward pass
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        else:
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / (1024 ** 2)  # MB
            
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_data)
            
        # Synchronize CUDA if using GPU
        if self.device == "cuda":
            torch.cuda.synchronize()
            
        # Measure memory after forward pass
        if self.device == "cuda":
            memory_after = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        else:
            process = psutil.Process(os.getpid())
            memory_after = process.memory_info().rss / (1024 ** 2)  # MB
            peak_memory = memory_after  # Not available for CPU
            
        # Calculate memory usage
        memory_used = memory_after - memory_before
        
        # Print results
        console.print(f"[green]✓[/green] Memory used: {memory_used:.2f} MB")
        if self.device == "cuda":
            console.print(f"[green]✓[/green] Peak memory: {peak_memory:.2f} MB")
            
        # Return memory metrics
        memory_metrics = {
            "memory_used_mb": memory_used,
            "peak_memory_mb": peak_memory if self.device == "cuda" else memory_after
        }
        
        return memory_metrics
        
    def benchmark(self, input_data: Optional[torch.Tensor] = None, 
                 warmup_runs: int = 5, 
                 timed_runs: int = 20) -> Dict[str, Any]:
        """Run a comprehensive benchmark of the model.
        
        Args:
            input_data: Input data to use for benchmarking. If None, dummy input is generated.
            warmup_runs: Number of warmup runs for latency measurement
            timed_runs: Number of timed runs for latency measurement
            
        Returns:
            Dictionary with benchmark results
        """
        # Generate dummy input if none provided
        if input_data is None:
            input_data = self.generate_dummy_input()
            
        # Measure latency
        latency = self.measure_latency(input_data, warmup_runs, timed_runs)
        
        # Measure memory
        memory_metrics = self.measure_memory(input_data)
        
        # Combine results
        results = {
            "latency_ms": latency * 1000,
            **memory_metrics
        }
        
        # Print summary
        table = Table(title="Benchmark Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Latency", f"{results['latency_ms']:.2f} ms")
        table.add_row("Memory Used", f"{results['memory_used_mb']:.2f} MB")
        if self.device == "cuda":
            table.add_row("Peak Memory", f"{results['peak_memory_mb']:.2f} MB")
            
        console.print(table)
        
        return results
        
    def save_results(self, results: Dict[str, Any], file_path: str) -> None:
        """Save benchmark results to a JSON file.
        
        Args:
            results: Benchmark results
            file_path: Path to save the results to
        """
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        console.print(f"[green]✓[/green] Results saved to {file_path}")
        
def benchmark_model(model: nn.Module, 
                   input_data: Optional[torch.Tensor] = None,
                   device: str = "cuda" if torch.cuda.is_available() else "cpu",
                   warmup_runs: int = 5,
                   timed_runs: int = 20,
                   save_path: Optional[str] = None) -> Dict[str, Any]:
    """Run a benchmark on a model.
    
    Args:
        model: PyTorch model to benchmark
        input_data: Input data to use for benchmarking. If None, dummy input is generated.
        device: Device to run the benchmark on
        warmup_runs: Number of warmup runs for latency measurement
        timed_runs: Number of timed runs for latency measurement
        save_path: Path to save the results to. If None, results are not saved.
        
    Returns:
        Dictionary with benchmark results
    """
    # Create benchmarker
    benchmarker = LLMBenchmarker(model, device)
    
    # Run benchmark
    results = benchmarker.benchmark(input_data, warmup_runs, timed_runs)
    
    # Save results if path provided
    if save_path:
        benchmarker.save_results(results, save_path)
        
    return results 