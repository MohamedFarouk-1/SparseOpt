"""
Command-line interface for SparseOpt.

This module provides a command-line interface for SparseOpt.
"""

import argparse
import json
import os
import torch
import torch.fx as fx
from typing import Dict, Any, Optional, List, Union
from rich.console import Console
from rich.table import Table

from .huggingface import optimize_hf_model, load_hf_model, create_model_input
from .utils.benchmark import benchmark_model, compare_models
from .optimize import optimize_model

console = Console()

def save_results(
    model: torch.nn.Module,
    stats: Dict[str, Any],
    output_dir: str,
    model_name: str
) -> None:
    """
    Save optimization results.
    
    Args:
        model: The optimized model
        stats: Optimization statistics
        output_dir: Directory to save results
        model_name: Name of the model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f"{model_name}_optimized.pt")
    torch.save(model.state_dict(), model_path)
    console.print(f"[bold green]Saved optimized model to {model_path}[/bold green]")
    
    # Save stats
    stats_path = os.path.join(output_dir, f"{model_name}_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    console.print(f"[bold green]Saved optimization stats to {stats_path}[/bold green]")
    
    # Save FX graph if available
    if hasattr(model, "graph"):
        graph_path = os.path.join(output_dir, f"{model_name}_graph.txt")
        with open(graph_path, "w") as f:
            f.write(str(model.graph))
        console.print(f"[bold green]Saved FX graph to {graph_path}[/bold green]")

def optimize_hf_model_cli(args: argparse.Namespace) -> None:
    """
    CLI entry point for optimizing HuggingFace models.
    
    Args:
        args: Command-line arguments
    """
    # Load model
    if args.demo:
        model_name = "gpt2"
        console.print(f"[bold blue]Running demo with {model_name}...[/bold blue]")
    else:
        model_name = args.hf_model
        console.print(f"[bold blue]Optimizing {model_name}...[/bold blue]")
    
    # Optimize model
    optimized_model, stats = optimize_hf_model(
        model_name,
        device=args.device,
        leaf_modules=args.leaf_modules
    )
    
    # Save results
    if args.output_dir:
        save_results(optimized_model, stats, args.output_dir, model_name)
    
    # Print summary
    table = Table(title="Optimization Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in stats.items():
        if isinstance(value, (int, float, str)):
            table.add_row(key, str(value))
    
    console.print(table)

def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="SparseOpt CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Optimize HuggingFace model
    hf_parser = subparsers.add_parser("optimize", help="Optimize a HuggingFace model")
    hf_parser.add_argument("--hf-model", type=str, help="HuggingFace model name")
    hf_parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="Device to run on")
    hf_parser.add_argument("--output-dir", type=str, help="Directory to save results")
    hf_parser.add_argument("--leaf-modules", type=str, nargs="+", help="Leaf modules to use")
    hf_parser.add_argument("--demo", action="store_true", help="Run demo with GPT-2")
    
    args = parser.parse_args()
    
    if args.command == "optimize":
        optimize_hf_model_cli(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 