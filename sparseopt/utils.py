"""Common utilities for SparseOpt."""

import torch
from typing import Any, Dict, List, Optional, Tuple
from rich.console import Console
from rich.table import Table
import time

console = Console()

def get_model_info(model: torch.nn.Module) -> Dict[str, Any]:
    """Get basic information about a PyTorch model.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Dictionary containing model information
    """
    return {
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "num_layers": len(list(model.modules())),
        "device": next(model.parameters()).device,
    }

def print_model_info(model: torch.nn.Module) -> None:
    """Print basic information about a PyTorch model.
    
    Args:
        model: PyTorch model to analyze
    """
    info = get_model_info(model)
    
    table = Table(title="Model Information")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Total Parameters", f"{info['num_parameters']:,}")
    table.add_row("Trainable Parameters", f"{info['num_trainable_parameters']:,}")
    table.add_row("Number of Layers", str(info['num_layers']))
    table.add_row("Device", str(info['device']))
    
    console.print(table)

def format_time(seconds: float) -> str:
    """Format time in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 1e-3:
        return f"{seconds*1e6:.2f} µs"
    elif seconds < 1:
        return f"{seconds*1e3:.2f} ms"
    else:
        return f"{seconds:.2f} s"

def print_summary_report(
    model_name: str,
    tracing_successful: bool,
    latency_before: float,
    latency_after: Optional[float] = None,
    optimizations_applied: List[str] = None
) -> None:
    """Print a summary report of the optimization results.
    
    Args:
        model_name: Name of the model
        tracing_successful: Whether FX tracing was successful
        latency_before: Latency before optimization (in seconds)
        latency_after: Latency after optimization (in seconds)
        optimizations_applied: List of optimizations applied
    """
    console.print("\n[bold]Optimization Summary Report[/bold]")
    console.print("=" * 50)
    
    # Model name
    console.print(f"[bold]Model:[/bold] {model_name}")
    
    # Tracing status
    if tracing_successful:
        console.print("[bold green]✓ FX Tracing:[/bold green] Successful")
    else:
        console.print("[bold red]✗ FX Tracing:[/bold red] Failed")
        console.print("[yellow]Note:[/yellow] Model was saved without optimization due to dynamic control flow.")
        return
    
    # Latency information
    console.print(f"[bold]Latency Before:[/bold] {format_time(latency_before)}")
    
    if latency_after is not None:
        console.print(f"[bold]Latency After:[/bold] {format_time(latency_after)}")
        
        # Calculate speedup
        speedup = latency_before / latency_after
        if speedup > 1:
            console.print(f"[bold green]Speedup:[/bold green] {speedup:.2f}x")
        else:
            console.print(f"[bold red]Slowdown:[/bold red] {1/speedup:.2f}x")
    
    # Optimizations applied
    if optimizations_applied:
        console.print("[bold]Optimizations Applied:[/bold]")
        for opt in optimizations_applied:
            console.print(f"  • {opt}")
    else:
        console.print("[bold]Optimizations Applied:[/bold] None")
    
    console.print("=" * 50) 