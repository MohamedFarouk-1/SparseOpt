#!/usr/bin/env python
"""
SparseOpt Colab Test Script

This script demonstrates how to use SparseOpt to optimize a HuggingFace model.
It's designed to be run in Google Colab.
"""

import argparse
import torch
import time
from rich.console import Console
from rich.table import Table

# Import SparseOpt
from sparseopt.huggingface import optimize_hf_model
from sparseopt.utils.benchmark import compare_models

console = Console()

def main():
    """Main entry point for the Colab test script."""
    parser = argparse.ArgumentParser(description="SparseOpt Colab Test")
    parser.add_argument("--hf-model", type=str, default="distilbert-base-uncased",
                        help="HuggingFace model name (default: distilbert-base-uncased)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], 
                        help="Device to run on (default: auto-detect)")
    parser.add_argument("--num-runs", type=int, default=100,
                        help="Number of benchmark runs (default: 100)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup runs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for inputs (default: 1)")
    parser.add_argument("--output-dir", type=str,
                        help="Directory to save results (optional)")
    
    args = parser.parse_args()
    
    # Auto-detect device if not provided
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    console.print(f"[bold blue]Running SparseOpt optimization on {args.hf_model}...[/bold blue]")
    console.print(f"[bold blue]Device: {args.device}[/bold blue]")
    
    # Optimize model
    start_time = time.time()
    optimized_model, stats = optimize_hf_model(
        args.hf_model,
        device=args.device
    )
    end_time = time.time()
    
    console.print(f"[bold green]Optimization completed in {end_time - start_time:.2f} seconds[/bold green]")
    
    # Print optimization stats
    table = Table(title="Optimization Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in stats.items():
        if isinstance(value, (int, float, str)):
            table.add_row(key, str(value))
    
    console.print(table)
    
    # Save results if output directory is provided
    if args.output_dir:
        import os
        import json
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(args.output_dir, f"{args.hf_model}_optimized.pt")
        torch.save(optimized_model.state_dict(), model_path)
        console.print(f"[bold green]Saved optimized model to {model_path}[/bold green]")
        
        # Save stats
        stats_path = os.path.join(args.output_dir, f"{args.hf_model}_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        console.print(f"[bold green]Saved optimization stats to {stats_path}[/bold green]")
    
    console.print("[bold green]SparseOpt optimization completed successfully![/bold green]")

if __name__ == "__main__":
    main() 