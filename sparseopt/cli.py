"""
Command Line Interface for SparseOpt.

This module provides a CLI for optimizing PyTorch models using the SparseOpt toolkit.
"""

import torch
import torch.fx as fx
from typing import Dict, Any
import argparse
import importlib.util
import sys
from pathlib import Path

from .graph.optimizer_pipeline import OptimizerPipeline
from .graph.passes.linear_fusion import LinearGELUFusion
from .graph.passes.dead_node import DeadNodeEliminationPass
from .graph.passes.reordering import NodeReorderingPass
from .graph.passes.conv_fusion import ConvBatchNormReLUFusion
from .huggingface import load_hf_model, create_hf_input, trace_hf_model, optimize_hf_model
from rich.console import Console

console = Console()

def load_model_from_file(model_file: str, model_class: str) -> torch.nn.Module:
    """Load a model class from a Python file."""
    # Get absolute path
    model_path = Path(model_file).resolve()
    
    # Load module
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_module"] = module
    spec.loader.exec_module(module)
    
    # Get model class
    if not hasattr(module, model_class):
        raise ValueError(f"Model class {model_class} not found in {model_file}")
    
    # Create model instance
    model_cls = getattr(module, model_class)
    return model_cls()

def create_example_inputs(input_shapes: Dict[str, tuple]) -> Dict[str, torch.Tensor]:
    """Create example inputs from shape specifications."""
    return {
        name: torch.randn(*shape)
        for name, shape in input_shapes.items()
    }

def main():
    parser = argparse.ArgumentParser(description="Optimize PyTorch models using SparseOpt")
    
    # Create argument groups
    model_group = parser.add_mutually_exclusive_group(required=True)
    
    # Custom model arguments
    custom_model = model_group.add_argument_group("Custom Model Options")
    custom_model.add_argument("--model-file", type=str,
                          help="Python file containing the model definition")
    custom_model.add_argument("--model-class", type=str,
                          help="Name of the model class to optimize")
    custom_model.add_argument("--input-shapes", type=str,
                          help="Input shapes as a comma-separated list of name:shape pairs. "
                               "Example: input:(1,3,224,224),mask:(1,1,224,224)")
    
    # HuggingFace model arguments
    hf_model = model_group.add_argument_group("HuggingFace Model Options")
    hf_model.add_argument("--hf-model", type=str,
                       help="HuggingFace model name (e.g., gpt2, distilbert-base-uncased)")
    hf_model.add_argument("--hf-model-type", type=str, default="auto",
                       help="Model type: auto, causal, sequence (default: auto)")
    hf_model.add_argument("--hf-task", type=str,
                       help="Task-specific model (e.g., text-generation, sequence-classification)")
    hf_model.add_argument("--text", type=str, default="Hello, this is SparseOpt!",
                       help="Input text for the model")
    hf_model.add_argument("--max-length", type=int, default=128,
                       help="Maximum sequence length")
    hf_model.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for inputs")
    
    # Optimization options
    parser.add_argument("--verify", action="store_true", default=True,
                      help="Verify numerical correctness after optimization")
    parser.add_argument("--benchmark", action="store_true", default=True,
                      help="Benchmark performance before and after optimization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to run optimization on (default: cuda if available, else cpu)")
    parser.add_argument("--save-model", type=str,
                      help="Path to save the optimized model (optional)")
    
    args = parser.parse_args()
    
    try:
        # Create optimizer pipeline
        optimizer = OptimizerPipeline(
            passes=[
                LinearGELUFusion(),
                DeadNodeEliminationPass(),
                NodeReorderingPass(),
                ConvBatchNormReLUFusion()
            ],
            verify_correctness=args.verify,
            benchmark=args.benchmark
        )
        
        if args.hf_model:
            # Load HuggingFace model
            console.print(f"[bold blue]Loading HuggingFace model {args.hf_model}...[/bold blue]")
            model, tokenizer = load_hf_model(
                model_name=args.hf_model,
                model_type=args.hf_model_type,
                device=args.device,
                task=args.hf_task
            )
            
            # Create inputs
            inputs = create_hf_input(
                tokenizer=tokenizer,
                text=args.text,
                device=args.device,
                max_length=args.max_length,
                batch_size=args.batch_size
            )
            
            # Run optimization
            optimized_model, stats = optimize_hf_model(
                model=model,
                tokenizer=tokenizer,
                optimizer_pipeline=optimizer,
                text=args.text,
                device=args.device
            )
            
        else:
            # Parse input shapes
            input_shapes = {}
            for pair in args.input_shapes.split(","):
                name, shape = pair.split(":")
                shape = tuple(int(x) for x in shape.strip("()").split("x"))
                input_shapes[name] = shape
            
            # Load custom model
            console.print(f"[bold blue]Loading model from {args.model_file}...[/bold blue]")
            model = load_model_from_file(args.model_file, args.model_class)
            model = model.to(args.device)
            model.eval()
            
            # Create example inputs
            example_inputs = create_example_inputs(input_shapes)
            example_inputs = {k: v.to(args.device) for k, v in example_inputs.items()}
            
            # Trace model
            console.print("[bold blue]Tracing model...[/bold blue]")
            traced_model = fx.symbolic_trace(model)
            
            # Run optimization
            optimized_model, stats = optimizer.optimize(traced_model, example_inputs)
        
        # Save optimized model if requested
        if args.save_model:
            console.print(f"[bold blue]Saving optimized model to {args.save_model}...[/bold blue]")
            torch.save(optimized_model.state_dict(), args.save_model)
        
        # Print results
        console.print("\n[bold green]Optimization Results:[/bold green]")
        for k, v in stats.items():
            if isinstance(v, float):
                console.print(f"• {k}: {v:.3f}")
            else:
                console.print(f"• {k}: {v}")
        
        console.print("[bold green]Optimization completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main() 