"""
Test script for optimizing HuggingFace models with SparseOpt.

This script tests the full optimization pipeline on real-world HuggingFace models
like GPT2 and DistilBERT, measuring performance improvements and verifying correctness.
"""

import torch
import torch.fx as fx
import argparse
from rich.console import Console
from rich.table import Table
import time

from sparseopt.graph.optimizer_pipeline import OptimizerPipeline
from sparseopt.graph.passes.linear_fusion import LinearGELUFusion
from sparseopt.graph.passes.dead_node import DeadNodeEliminationPass
from sparseopt.graph.passes.reordering import NodeReorderingPass
from sparseopt.graph.passes.conv_fusion import ConvBatchNormReLUFusion
from sparseopt.huggingface import load_hf_model, create_hf_input, trace_hf_model, optimize_hf_model

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Test SparseOpt on HuggingFace models")
    
    # Model selection
    parser.add_argument("--model", type=str, default="gpt2", 
                      help="HuggingFace model name (default: gpt2)")
    parser.add_argument("--model-type", type=str, default="auto",
                      help="Model type: auto, causal, sequence (default: auto)")
    parser.add_argument("--task", type=str, default=None,
                      help="Task-specific model (e.g., text-generation, sequence-classification)")
    
    # Input configuration
    parser.add_argument("--text", type=str, default="Hello, this is SparseOpt!",
                      help="Input text for the model")
    parser.add_argument("--max-length", type=int, default=128,
                      help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=1,
                      help="Batch size for inputs")
    
    # Optimization options
    parser.add_argument("--verify", action="store_true", default=True,
                      help="Verify numerical correctness after optimization")
    parser.add_argument("--benchmark", action="store_true", default=True,
                      help="Benchmark performance before and after optimization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to run optimization on (default: cuda if available, else cpu)")
    
    # Pass selection
    parser.add_argument("--passes", type=str, default="all",
                      help="Comma-separated list of passes to apply (default: all)")
    
    return parser.parse_args()

def get_optimization_passes(pass_names):
    """Get the list of optimization passes to apply."""
    all_passes = {
        "linear_gelu": LinearGELUFusion(),
        "dead_node": DeadNodeEliminationPass(),
        "reordering": NodeReorderingPass(),
        "conv_bn_relu": ConvBatchNormReLUFusion()
    }
    
    if pass_names == "all":
        return list(all_passes.values())
    
    passes = []
    for name in pass_names.split(","):
        name = name.strip().lower()
        if name in all_passes:
            passes.append(all_passes[name])
        else:
            console.print(f"[yellow]Warning: Unknown pass '{name}', skipping[/yellow]")
    
    return passes

def main():
    args = parse_args()
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_hf_model(
            model_name=args.model,
            model_type=args.model_type,
            device=args.device,
            task=args.task
        )
        
        # Create inputs
        inputs = create_hf_input(
            tokenizer=tokenizer,
            text=args.text,
            device=args.device,
            max_length=args.max_length,
            batch_size=args.batch_size
        )
        
        # Get optimization passes
        passes = get_optimization_passes(args.passes)
        
        # Create optimizer pipeline
        optimizer = OptimizerPipeline(
            passes=passes,
            verify_correctness=args.verify,
            benchmark=args.benchmark
        )
        
        # Run optimization
        optimized_model, stats = optimize_hf_model(
            model=model,
            tokenizer=tokenizer,
            optimizer_pipeline=optimizer,
            text=args.text,
            device=args.device,
            max_length=args.max_length,
            batch_size=args.batch_size
        )
        
        # Print final summary
        console.print("\n[bold green]Optimization completed successfully![/bold green]")
        
        # Print model info
        model_info = {
            "Model": args.model,
            "Parameters": f"{sum(p.numel() for p in model.parameters()):,}",
            "Layers": len(model.transformer.h) if hasattr(model, "transformer") else "N/A"
        }
        
        info_table = Table(title="Model Information")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="magenta")
        
        for k, v in model_info.items():
            info_table.add_row(k, str(v))
        
        console.print(info_table)
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise

if __name__ == "__main__":
    main() 