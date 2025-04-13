#!/usr/bin/env python
"""Example script demonstrating how to use SparseOpt with MoE models."""

import argparse
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparseopt.optimize import ModelOptimizer
from sparseopt.benchmark_llm import benchmark_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Optimize a MoE model with SparseOpt")
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-v0.1", 
                        help="HuggingFace model name or path")
    parser.add_argument("--input-text", type=str, default="Hello, world!", 
                        help="Input text for benchmarking")
    parser.add_argument("--max-length", type=int, default=128, 
                        help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=1, 
                        help="Batch size for benchmarking")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run the benchmark on")
    parser.add_argument("--warmup-runs", type=int, default=5, 
                        help="Number of warmup runs for benchmarking")
    parser.add_argument("--timed-runs", type=int, default=20, 
                        help="Number of timed runs for benchmarking")
    parser.add_argument("--save-report", type=str, default=None, 
                        help="Path to save the benchmark report")
    parser.add_argument("--use-layer-by-layer", action="store_true", 
                        help="Use layer-by-layer optimization")
    parser.add_argument("--skip-layers", type=str, nargs="+", default=[], 
                        help="Layers to skip optimization")
    parser.add_argument("--fusion-patterns", type=str, nargs="+", 
                        default=["linear_relu", "linear_gelu", "layer_norm"], 
                        help="Fusion patterns to apply")
    parser.add_argument("--num-experts", type=int, default=8, 
                        help="Number of experts to use (for models that support it)")
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Load model and tokenizer
    console.print(f"[bold]Loading model: {args.model}[/bold]")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Set number of experts if supported
    if hasattr(model, "set_num_experts"):
        console.print(f"[bold]Setting number of experts to {args.num_experts}[/bold]")
        model.set_num_experts(args.num_experts)
    
    # Prepare input
    console.print(f"[bold]Preparing input: {args.input_text}[/bold]")
    inputs = tokenizer(args.input_text, return_tensors="pt", max_length=args.max_length, truncation=True)
    input_ids = inputs["input_ids"].to(args.device)
    
    # Benchmark before optimization
    console.print("[bold]Benchmarking before optimization[/bold]")
    before_results = benchmark_model(
        model, 
        input_data=input_ids,
        device=args.device,
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
        save_path=f"{args.save_report}_before.json" if args.save_report else None
    )
    
    # Optimize model
    console.print("[bold]Optimizing model[/bold]")
    optimizer = ModelOptimizer(model)
    optimized_model = optimizer.optimize(
        use_layer_by_layer=args.use_layer_by_layer,
        skip_layers=args.skip_layers,
        fusion_patterns=args.fusion_patterns
    )
    
    # Benchmark after optimization
    console.print("[bold]Benchmarking after optimization[/bold]")
    after_results = benchmark_model(
        optimized_model, 
        input_data=input_ids,
        device=args.device,
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
        save_path=f"{args.save_report}_after.json" if args.save_report else None
    )
    
    # Print comparison
    console.print("[bold]Comparison[/bold]")
    table = Table(title="Optimization Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Before", style="magenta")
    table.add_column("After", style="green")
    table.add_column("Improvement", style="yellow")
    
    # Latency comparison
    latency_before = before_results["latency_ms"]
    latency_after = after_results["latency_ms"]
    latency_improvement = (latency_before - latency_after) / latency_before * 100
    
    table.add_row("Latency (ms)", f"{latency_before:.2f}", f"{latency_after:.2f}", f"{latency_improvement:.2f}%")
    
    # Memory comparison
    memory_before = before_results["memory_used_mb"]
    memory_after = after_results["memory_used_mb"]
    memory_improvement = (memory_before - memory_after) / memory_before * 100
    
    table.add_row("Memory Used (MB)", f"{memory_before:.2f}", f"{memory_after:.2f}", f"{memory_improvement:.2f}%")
    
    # Peak memory comparison (if available)
    if "peak_memory_mb" in before_results and "peak_memory_mb" in after_results:
        peak_memory_before = before_results["peak_memory_mb"]
        peak_memory_after = after_results["peak_memory_mb"]
        peak_memory_improvement = (peak_memory_before - peak_memory_after) / peak_memory_before * 100
        
        table.add_row("Peak Memory (MB)", f"{peak_memory_before:.2f}", f"{peak_memory_after:.2f}", f"{peak_memory_improvement:.2f}%")
    
    console.print(table)
    
    # Save combined report if requested
    if args.save_report:
        combined_results = {
            "model": args.model,
            "num_experts": args.num_experts,
            "before": before_results,
            "after": after_results,
            "improvement": {
                "latency": latency_improvement,
                "memory": memory_improvement
            }
        }
        
        with open(f"{args.save_report}_combined.json", 'w') as f:
            json.dump(combined_results, f, indent=2)
            
        console.print(f"[green]✓[/green] Combined report saved to {args.save_report}_combined.json")

if __name__ == "__main__":
    from rich.console import Console
    import json
    from rich.table import Table
    
    console = Console()
    main() 