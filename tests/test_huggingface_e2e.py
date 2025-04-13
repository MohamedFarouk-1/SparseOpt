#!/usr/bin/env python
"""
End-to-end test for HuggingFace model optimization in SparseOpt.
This script validates the entire optimization pipeline:
1. Model loading and tracing
2. Pattern fusion
3. Numerical correctness
4. Performance benchmarking
5. CLI output formatting
"""

import os
import sys
import torch
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Add parent directory to path to import sparseopt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sparseopt.huggingface import (
    load_huggingface_model,
    create_model_input,
    trace_model,
    benchmark_model,
    print_benchmark_results
)
from sparseopt.graph.optimizer import GraphOptimizer
from sparseopt.graph.passes import LinearGELUFusion, ConvBatchNormReLUFusion

console = Console()

def run_end_to_end_test(
    model_name: str = "gpt2",
    sample_text: str = "Hello, this is SparseOpt!",
    device: str = None,
    num_runs: int = 100,
    warmup: int = 10,
    rtol: float = 1e-3,
    atol: float = 1e-3
):
    """
    Run a complete end-to-end test of the HuggingFace optimization pipeline.
    
    Args:
        model_name: Name of the HuggingFace model to test
        sample_text: Sample text for input
        device: Device to run on (default: cuda if available, else cpu)
        num_runs: Number of benchmark runs
        warmup: Number of warmup runs
        rtol: Relative tolerance for numerical correctness
        atol: Absolute tolerance for numerical correctness
        
    Returns:
        True if all checks passed, False otherwise
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    console.print(Panel(f"Running end-to-end test for {model_name} on {device.upper()}", 
                        title="SparseOpt HuggingFace Test", 
                        border_style="blue"))
    
    # Step 1: Load model and tokenizer
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Loading model...", total=100)
        
        try:
            model, tokenizer, model_info = load_huggingface_model(model_name, device)
            progress.update(task, completed=100)
        except Exception as e:
            console.print(f"[red]❌ Failed to load model: {str(e)}[/red]")
            return False
    
    # Print model info
    model_table = Table(title="Model Information")
    model_table.add_column("Property", style="cyan")
    model_table.add_column("Value", style="green")
    
    model_table.add_row("Model", model_name)
    model_table.add_row("Parameters", f"{model_info['num_parameters']:,}")
    model_table.add_row("Layers", str(model_info['num_layers']))
    model_table.add_row("Device", device.upper())
    
    console.print(model_table)
    
    # Step 2: Create input tensors
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Creating input tensors...", total=100)
        
        try:
            inputs = create_model_input(tokenizer, sample_text)
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            progress.update(task, completed=100)
        except Exception as e:
            console.print(f"[red]❌ Failed to create input tensors: {str(e)}[/red]")
            return False
    
    # Step 3: Trace model
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Tracing model...", total=100)
        
        try:
            traced_model, tracing_info = trace_model(model, inputs)
            if not tracing_info["success"]:
                console.print(f"[red]❌ Failed to trace model: {tracing_info.get('error', 'Unknown error')}[/red]")
                return False
            progress.update(task, completed=100)
        except Exception as e:
            console.print(f"[red]❌ Failed to trace model: {str(e)}[/red]")
            return False
    
    # Print tracing info
    console.print(f"[green]✓ Model traced successfully with {tracing_info['num_nodes']} nodes[/green]")
    
    # Step 4: Apply optimization
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Applying optimization...", total=100)
        
        try:
            # Create optimizer with passes
            optimizer = GraphOptimizer([
                LinearGELUFusion(),
                ConvBatchNormReLUFusion()
            ])
            
            # Apply optimization
            optimized_model, fusion_stats = optimizer.optimize(traced_model, inputs)
            progress.update(task, completed=100)
        except Exception as e:
            console.print(f"[red]❌ Failed to apply optimization: {str(e)}[/red]")
            return False
    
    # Check if any fusions were applied
    num_fusions = fusion_stats.get("num_fusions", 0)
    if num_fusions == 0:
        console.print("[yellow]⚠️ No fusion patterns were found in the model[/yellow]")
    else:
        console.print(f"[green]✓ Applied {num_fusions} fusion patterns[/green]")
    
    # Step 5: Verify numerical correctness
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Verifying numerical correctness...", total=100)
        
        try:
            with torch.no_grad():
                original_output = model(**inputs)
                optimized_output = optimized_model(**inputs)
                
                # Check that outputs are close
                is_correct = torch.allclose(
                    original_output.last_hidden_state,
                    optimized_output.last_hidden_state,
                    rtol=rtol,
                    atol=atol
                )
                
                if is_correct:
                    console.print("[green]✓ Numerical correctness verified[/green]")
                    progress.update(task, completed=100)
                else:
                    console.print("[red]❌ Numerical correctness check failed[/red]")
                    return False
        except Exception as e:
            console.print(f"[red]❌ Failed to verify numerical correctness: {str(e)}[/red]")
            return False
    
    # Step 6: Benchmark models
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Benchmarking models...", total=100)
        
        try:
            # Benchmark original model
            original_results = benchmark_model(
                model, 
                inputs, 
                device=device,
                num_runs=num_runs,
                warmup=warmup
            )
            
            # Benchmark optimized model
            optimized_results = benchmark_model(
                optimized_model, 
                inputs, 
                device=device,
                num_runs=num_runs,
                warmup=warmup
            )
            
            progress.update(task, completed=100)
        except Exception as e:
            console.print(f"[red]❌ Failed to benchmark models: {str(e)}[/red]")
            return False
    
    # Print benchmark results
    print_benchmark_results(
        original_results,
        optimized_results,
        fusion_stats
    )
    
    # Calculate speedup
    speedup = original_results["mean_latency"] / optimized_results["mean_latency"]
    
    # Print final summary
    console.print(Panel(
        f"✅ All checks passed!\n\n"
        f"Model: {model_name}\n"
        f"Device: {device.upper()}\n"
        f"Fusion patterns: {num_fusions}\n"
        f"Speedup: {speedup:.2f}x",
        title="Test Summary",
        border_style="green"
    ))
    
    return True

def main():
    """Main entry point for the end-to-end test."""
    parser = argparse.ArgumentParser(description="End-to-end test for HuggingFace model optimization")
    parser.add_argument("--model", type=str, default="gpt2", help="HuggingFace model name")
    parser.add_argument("--text", type=str, default="Hello, this is SparseOpt!", help="Sample text for input")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda/cpu)")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for numerical correctness")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for numerical correctness")
    
    args = parser.parse_args()
    
    success = run_end_to_end_test(
        model_name=args.model,
        sample_text=args.text,
        device=args.device,
        num_runs=args.num_runs,
        warmup=args.warmup,
        rtol=args.rtol,
        atol=args.atol
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 