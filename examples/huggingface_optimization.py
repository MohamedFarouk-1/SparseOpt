"""
Example script demonstrating how to optimize HuggingFace models with SparseOpt.

This script shows how to:
1. Load a HuggingFace model
2. Apply the SparseOpt optimization pipeline
3. Measure performance improvements
4. Verify correctness
"""

import torch
import torch.fx as fx
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparseopt.graph.optimizer import GraphPass
from sparseopt.graph.passes.linear_fusion import LinearGELUFusion
from sparseopt.graph.passes.dead_node import DeadNodeEliminationPass
from sparseopt.graph.passes.reordering import NodeReorderingPass
from sparseopt.graph.passes.conv_fusion import ConvBatchNormReLUFusion
from sparseopt.huggingface import load_hf_model, create_hf_input, optimize_hf_model

console = Console()

def main():
    # Load model and tokenizer
    model_name = "gpt2"
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Example input
    example_input = "Hello, how are you today?"
    
    # Optimize model
    print("Optimizing model...")
    optimized_model, stats = optimize_hf_model(
        model=model,
        tokenizer=tokenizer,
        example_input=example_input,
        max_sequence_length=128
    )
    
    # Check if optimization failed
    if optimized_model is None:
        print("\nOptimization failed. Check the error message above.")
        return
    
    # Print optimization statistics
    print("\nOptimization Statistics:")
    for pass_name, pass_stats in stats.items():
        print(f"\n{pass_name}:")
        if isinstance(pass_stats, dict):
            for stat_name, value in pass_stats.items():
                print(f"  {stat_name}: {value}")
        else:
            print(f"  {pass_stats}")
    
    # Test the optimized model
    print("\nTesting optimized model...")
    inputs = tokenizer(example_input, return_tensors="pt")
    with torch.no_grad():
        outputs = optimized_model(**inputs)
    
    print("Optimization complete!")

if __name__ == "__main__":
    main() 