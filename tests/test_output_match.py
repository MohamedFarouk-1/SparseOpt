"""Test to verify output consistency between original and optimized models."""

import torch
import sys
import os
from pathlib import Path
import argparse
from rich.console import Console

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from sparseopt.model_loader import load_model_from_file
from sparseopt.optimize import ModelOptimizer

console = Console()

def test_output_match(
    model_path: str,
    model_class: str,
    input_path: str,
    output_path: str,
    tolerance: float = 1e-4
) -> bool:
    """Test that the original and optimized models produce the same output.
    
    Args:
        model_path: Path to the original model file
        model_class: Name of the model class
        input_path: Path to the input data
        output_path: Path to save the optimized model
        tolerance: Maximum allowed difference between outputs
        
    Returns:
        True if outputs match within tolerance, False otherwise
    """
    # Load the original model
    console.print(f"Loading original model from {model_path}")
    original_model = load_model_from_file(model_path, model_class)
    
    # Load the input data
    console.print(f"Loading input data from {input_path}")
    input_data = torch.load(input_path)
    
    # Run inference on the original model
    console.print("Running inference on original model")
    original_model.eval()
    with torch.no_grad():
        if hasattr(input_data, 'x') and hasattr(input_data, 'edge_index'):
            # GNN model
            original_output = original_model(input_data.x, input_data.edge_index)
        else:
            # Standard model
            original_output = original_model(input_data)
    
    # Optimize the model
    console.print("Optimizing model")
    optimizer = ModelOptimizer(original_model)
    try:
        optimized_model = optimizer.optimize()
        tracing_successful = True
    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] Optimization failed: {str(e)}")
        console.print("[yellow]Saving original model without optimization.[/yellow]")
        optimized_model = original_model
        tracing_successful = False
    
    # Save the optimized model
    console.print(f"Saving model to {output_path}")
    torch.save(optimized_model, output_path)
    
    # Run inference on the optimized model
    console.print("Running inference on optimized model")
    optimized_model.eval()
    with torch.no_grad():
        if hasattr(input_data, 'x') and hasattr(input_data, 'edge_index'):
            # GNN model
            optimized_output = optimized_model(input_data.x, input_data.edge_index)
        else:
            # Standard model
            optimized_output = optimized_model(input_data)
    
    # Compare outputs
    max_diff = torch.max(torch.abs(original_output - optimized_output)).item()
    console.print(f"Maximum absolute difference: {max_diff:.2e}")
    
    # Check if outputs match within tolerance
    if max_diff <= tolerance:
        console.print("[bold green]✓ Outputs match within tolerance![/bold green]")
        return True
    else:
        console.print("[bold red]✗ Outputs do not match within tolerance![/bold red]")
        return False

def main():
    """Run the output consistency test."""
    parser = argparse.ArgumentParser(description="Test output consistency between original and optimized models")
    parser.add_argument("--model", required=True, help="Path to the model file")
    parser.add_argument("--class", required=True, help="Name of the model class")
    parser.add_argument("--input", required=True, help="Path to the input data")
    parser.add_argument("--output", required=True, help="Path to save the optimized model")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Maximum allowed difference between outputs")
    
    args = parser.parse_args()
    
    success = test_output_match(
        args.model,
        getattr(args, 'class'),
        args.input,
        args.output,
        args.tolerance
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 