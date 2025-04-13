"""
Optimizer Pipeline.

This module provides a unified pipeline for applying multiple optimization passes
to a PyTorch model, collecting statistics, and verifying correctness.
"""

import torch
import torch.fx as fx
from typing import List, Tuple, Dict, Any, Optional
from rich.console import Console
from rich.table import Table
import time

class OptimizerPipeline:
    """
    A pipeline that applies multiple optimization passes to a PyTorch model.
    
    The pipeline manages the application of passes, collects statistics,
    verifies correctness, and optionally benchmarks performance.
    """
    
    def __init__(self, passes: List, verify_correctness: bool = True, benchmark: bool = True):
        """
        Initialize the optimizer pipeline.
        
        Args:
            passes: List of optimization passes to apply
            verify_correctness: Whether to verify numerical correctness after each pass
            benchmark: Whether to benchmark performance before and after optimization
        """
        self.passes = passes
        self.verify_correctness = verify_correctness
        self.benchmark = benchmark
        self.console = Console()
        
    def optimize(self, graph_module: fx.GraphModule, example_inputs: Dict[str, torch.Tensor]) -> Tuple[fx.GraphModule, Dict[str, Any]]:
        """
        Apply optimization passes to the graph module.
        
        Args:
            graph_module: The graph module to optimize
            example_inputs: Example inputs for tracing and verification
            
        Returns:
            Tuple of (optimized graph module, statistics)
        """
        stats = {}
        gm = graph_module
        
        # Store original outputs for verification
        if self.verify_correctness:
            with torch.no_grad():
                original_output = gm(**example_inputs)
        
        # Measure original latency if benchmarking
        if self.benchmark:
            original_latency = self._measure_latency(gm, example_inputs)
            stats["original_latency_ms"] = original_latency
        
        # Apply each pass in sequence
        for i, pass_ in enumerate(self.passes):
            pass_name = pass_.__class__.__name__
            self.console.print(f"\n[bold blue]Applying {pass_name}...[/bold blue]")
            
            # Apply the pass
            gm, pass_stats = pass_.apply(gm, example_inputs)
            
            # Update stats with pass-specific metrics
            for k, v in pass_stats.items():
                key = f"{pass_name.lower()}_{k}"
                stats[key] = v
            
            # Verify correctness if enabled
            if self.verify_correctness:
                with torch.no_grad():
                    current_output = gm(**example_inputs)
                    is_correct = torch.allclose(original_output, current_output, rtol=1e-4, atol=1e-4)
                    stats[f"{pass_name.lower()}_correct"] = is_correct
                    if not is_correct:
                        self.console.print(f"[bold red]Warning: {pass_name} produced incorrect results![/bold red]")
        
        # Measure final latency if benchmarking
        if self.benchmark:
            final_latency = self._measure_latency(gm, example_inputs)
            stats["optimized_latency_ms"] = final_latency
            stats["speedup"] = original_latency / final_latency
        
        # Print optimization summary
        self._print_summary(stats)
        
        return gm, stats
    
    def _measure_latency(self, model: fx.GraphModule, inputs: Dict[str, torch.Tensor], num_warmup: int = 10, num_iter: int = 100) -> float:
        """Measure model latency in milliseconds."""
        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                model(**inputs)
        
        # Time iterations
        start_time = time.perf_counter()
        for _ in range(num_iter):
            with torch.no_grad():
                model(**inputs)
        end_time = time.perf_counter()
        
        # Return average latency in milliseconds
        return ((end_time - start_time) / num_iter) * 1000
    
    def _print_summary(self, stats: Dict[str, Any]) -> None:
        """Print a rich summary table of optimization results."""
        table = Table(title="Optimization Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        # Add pass-specific stats
        for pass_ in self.passes:
            pass_name = pass_.__class__.__name__.lower()
            for k, v in stats.items():
                if k.startswith(pass_name):
                    metric = k.replace(f"{pass_name}_", "").replace("_", " ").title()
                    if isinstance(v, bool):
                        value = "✓" if v else "✗"
                    elif isinstance(v, float):
                        value = f"{v:.3f}"
                    else:
                        value = str(v)
                    table.add_row(metric, value)
        
        # Add performance metrics if available
        if "original_latency_ms" in stats:
            table.add_row("Original Latency (ms)", f"{stats['original_latency_ms']:.3f}")
            table.add_row("Optimized Latency (ms)", f"{stats['optimized_latency_ms']:.3f}")
            table.add_row("Speedup", f"{stats['speedup']:.2f}x")
        
        self.console.print(table) 