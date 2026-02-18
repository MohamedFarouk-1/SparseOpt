"""
Demo model definitions and loader for SparseOpt CLI.

Provides three ready-to-go models that cover the major optimization targets:
  resnet50  — CNN with Conv+BN+ReLU fusion opportunities
  bert-base — Transformer with Linear+GELU and Linear+LayerNorm patterns
  mlp       — Simple MLP with Linear+ReLU and Dropout elimination
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Built-in demo model
# ---------------------------------------------------------------------------

class DemoMLP(nn.Module):
    """Simple MLP with dropout — demonstrates Linear+ReLU fusion and
    dropout elimination.

    Architecture: Linear(512→2048) → ReLU → Dropout(0.1)
                  → Linear(2048→2048) → ReLU → Dropout(0.1)
                  → Linear(2048→512) → ReLU
                  → Linear(512→10)
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(512, 2048)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(2048, 2048)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(2048, 512)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.fc1(x))
        x = self.drop1(x)
        x = self.relu2(self.fc2(x))
        x = self.drop2(x)
        x = self.relu3(self.fc3(x))
        return self.fc4(x)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def get_demo_model(
    name: str,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
    """Return an (eval-mode model, example_inputs) pair for a named demo model.

    Args:
        name:   One of "resnet50", "bert-base", "mlp".
        device: Target device string, e.g. "cpu" or "cuda".

    Returns:
        (model, example_inputs) where example_inputs keys match the model's
        forward() parameter names.

    Raises:
        ValueError: For unknown model names.
        ImportError: If torchvision/transformers are not installed.
    """
    if name == "resnet50":
        model, inputs = _load_resnet50()
    elif name == "bert-base":
        model, inputs = _load_bert_base()
    elif name == "mlp":
        model, inputs = _load_mlp()
    else:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: resnet50, bert-base, mlp"
        )

    model = model.to(device).eval()
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return model, inputs


def _load_resnet50() -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
    try:
        import torchvision.models as tvm
    except ImportError:
        raise ImportError(
            "torchvision is required for the resnet50 demo. "
            "Install it with: pip install torchvision"
        )
    model = tvm.resnet50(weights=None)
    inputs = {"x": torch.randn(1, 3, 224, 224)}
    return model, inputs


def _load_bert_base() -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
    try:
        from transformers import AutoModel
    except ImportError:
        raise ImportError(
            "transformers is required for the bert-base demo. "
            "Install it with: pip install transformers"
        )
    model = AutoModel.from_pretrained("bert-base-uncased")
    inputs = {
        "input_ids": torch.randint(0, 30522, (1, 64)),
        "attention_mask": torch.ones(1, 64, dtype=torch.long),
    }
    return model, inputs


def _load_mlp() -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
    model = DemoMLP()
    inputs = {"x": torch.randn(1, 512)}
    return model, inputs
