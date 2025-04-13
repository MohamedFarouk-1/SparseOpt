"""Error handling utilities for SparseOpt."""

import re
from typing import Optional, Dict, Any, List, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

class SparseOptError(Exception):
    """Base exception for SparseOpt errors."""
    pass

class SymbolicConditionalError(SparseOptError):
    """Exception raised when a model contains symbolic tensor conditionals."""
    pass

class ModelFileNotFoundError(SparseOptError):
    """Exception raised when a model file cannot be found."""
    pass

class ModelClassNotFoundError(SparseOptError):
    """Exception raised when a model class cannot be found in a file."""
    pass

class InputShapeMismatchError(SparseOptError):
    """Exception raised when input shape doesn't match model's expected shape."""
    pass

class TorchFxTracingError(SparseOptError):
    """Exception raised when Torch.fx tracing fails."""
    pass

# Error message templates
ERROR_MESSAGES = {
    SymbolicConditionalError: {
        "title": "Symbolic Conditional Error",
        "description": "Torch.fx cannot trace conditionals based on tensor values.",
        "example": "if torch.mean(x) > 0:",
        "solution": "Refactor your model to avoid conditionals based on tensor values. Use control flow based on Python values instead."
    },
    ModelFileNotFoundError: {
        "title": "Model File Not Found",
        "description": "The specified model file could not be found.",
        "solution": "Check that the file path is correct and the file exists."
    },
    ModelClassNotFoundError: {
        "title": "Model Class Not Found",
        "description": "The specified model class could not be found in the file.",
        "solution": "Check that the class name is correct and exists in the file."
    },
    InputShapeMismatchError: {
        "title": "Input Shape Mismatch",
        "description": "The provided input shape doesn't match what the model expects.",
        "solution": "Adjust the input shape to match the model's expected input dimensions."
    },
    TorchFxTracingError: {
        "title": "Torch.fx Tracing Error",
        "description": "Failed to trace the model with Torch.fx.",
        "solution": "Check for unsupported operations or patterns in your model."
    }
}

def detect_error_type(error: Exception) -> Tuple[type, Optional[str]]:
    """Detect the type of error and extract relevant details.
    
    Args:
        error: The exception to analyze
        
    Returns:
        Tuple of (error_type, error_details)
    """
    error_msg = str(error).lower()
    
    # Check for symbolic conditional errors
    if "symbolic" in error_msg or "tensor" in error_msg:
        if re.search(r"if.*tensor.*>|<|==|!=|>=|<=", error_msg, re.IGNORECASE):
            return SymbolicConditionalError, error_msg
            
    # Check for file not found errors
    if "file" in error_msg and ("not found" in error_msg or "does not exist" in error_msg):
        return ModelFileNotFoundError, error_msg
        
    # Check for class not found errors
    if "class" in error_msg and "not found" in error_msg:
        return ModelClassNotFoundError, error_msg
        
    # Check for shape mismatch errors
    if "shape" in error_msg and ("mismatch" in error_msg or "cannot be multiplied" in error_msg):
        return InputShapeMismatchError, error_msg
        
    # Check for Torch.fx tracing errors
    if "torch.fx" in error_msg or "tracing" in error_msg:
        return TorchFxTracingError, error_msg
        
    # Default to generic SparseOptError
    return SparseOptError, error_msg

def print_error(error: Exception, debug: bool = False) -> None:
    """Print a formatted error message.
    
    Args:
        error: The exception to display
        debug: Whether to show the full traceback
    """
    error_type, error_details = detect_error_type(error)
    
    # Get error message template
    template = ERROR_MESSAGES.get(error_type, {
        "title": "Error",
        "description": str(error),
        "solution": "Check the error message for details."
    })
    
    # Print error panel
    console.print(Panel(
        f"[bold red]{template['title']}[/bold red]\n\n"
        f"[yellow]{template['description']}[/yellow]\n\n"
        f"[green]Solution:[/green] {template['solution']}\n\n"
        f"[blue]Example:[/blue] {template.get('example', 'N/A')}",
        title="SparseOpt Error",
        border_style="red"
    ))
    
    # Print debug information if requested
    if debug:
        console.print("\n[bold]Debug Information:[/bold]")
        console.print(Syntax(str(error), "python", theme="monokai"))
        import traceback
        console.print("\n[bold]Traceback:[/bold]")
        console.print(Syntax("".join(traceback.format_tb(error.__traceback__)), "python", theme="monokai"))

def validate_input_shape(model: Any, input_shape: Tuple[int, ...]) -> None:
    """Validate that the input shape matches the model's expected shape.
    
    Args:
        model: The PyTorch model
        input_shape: The input shape to validate
        
    Raises:
        InputShapeMismatchError: If the input shape doesn't match the model's expected shape
    """
    # This is a simplified validation - in a real implementation,
    # you would inspect the model's first layer to determine the expected input shape
    try:
        # Try to create a dummy input with the given shape
        import torch
        dummy_input = torch.randn(input_shape)
        model(dummy_input)
    except RuntimeError as e:
        error_msg = str(e)
        if "shape" in error_msg and ("mismatch" in error_msg or "cannot be multiplied" in error_msg):
            raise InputShapeMismatchError(
                f"The input shape {input_shape} doesn't match what the model expects. "
                f"Error: {error_msg}"
            )
        # Re-raise other runtime errors
        raise 