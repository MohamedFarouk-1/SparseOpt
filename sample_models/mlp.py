import torch
import torch.nn as nn

# Define a simple MLP model that works perfectly with torch.fx
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.seq(x)

# Make the class importable
__all__ = ["SimpleMLP"]

# Save model and example input
if __name__ == "__main__":
    model = SimpleMLP()
    x = torch.randn((5, 16))

    torch.save(model.state_dict(), "mlp.pt")
    torch.save(x, "mlp_input.pt")

    print("✅ Saved model to mlp.pt")
    print("✅ Saved input to mlp_input.pt") 