"""Model loading utilities for SparseOpt."""

import torch
import torch.nn as nn
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Type, Tuple

from .errors import ModelFileNotFoundError, ModelClassNotFoundError

# Map of model class names to their implementations
MODEL_CLASS_MAP = {
    "SimpleGCN": None,  # Will be populated dynamically
    "GCNModel": None,   # Will be populated dynamically
}

def register_model_class(name: str, model_class: Type[nn.Module]) -> None:
    """Register a model class for loading.
    
    Args:
        name: Name of the model class
        model_class: The model class to register
    """
    MODEL_CLASS_MAP[name] = model_class

def load_model_from_file(path: str, model_class: str):
    print(f"✅ Using load_model_from_file with class: {model_class}")

    from sample_models.mlp import SimpleMLP
    from sample_models.gcn import SimpleGCN
    import torch

    model_class_map = {
        "SimpleMLP": SimpleMLP,
        "SimpleGCN": SimpleGCN
    }

    if model_class not in model_class_map:
        raise ValueError(f"Unknown model class: {model_class}")

    model = model_class_map[model_class]()
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def generate_model_code(model: nn.Module) -> str:
    """Generate Python code for a PyTorch model.
    
    Args:
        model: PyTorch model to generate code for
        
    Returns:
        String containing Python code for the model
    """
    code_lines = []
    
    # Add model attributes
    for name, module in model.named_children():
        if isinstance(module, nn.Module):
            code_lines.append(f"self.{name} = {module.__class__.__name__}()")
            
    return "\n        ".join(code_lines)

def save_model(
    model: nn.Module,
    save_path: str,
    model_name: str = "model"
) -> None:
    """Save a PyTorch model to a file.
    
    Args:
        model: PyTorch model to save
        save_path: Path where to save the model
        model_name: Name of the model class
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the model state dict
    torch.save(model.state_dict(), save_path)
    
    # Create a Python file with the model definition
    model_file = save_path.with_suffix('.py')
    with open(model_file, "w") as f:
        f.write(f"""import torch
import torch.nn as nn

class {model_name}(nn.Module):
    def __init__(self):
        super().__init__()
        {generate_model_code(model)}
        
    def forward(self, x):
        return self.model(x)
""") 