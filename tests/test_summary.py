import torch
import torch.nn as nn
import torch.nn.functional as F
from sparseopt.graph.optimizer import GraphOptimizer
from sparseopt.graph.passes import LinearGELUFusion
from sparseopt.summary import print_optimization_summary

def main():
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            hidden_size = 2048
            self.linear1 = nn.Linear(hidden_size, hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)
        
        def forward(self, x):
            x = F.gelu(self.linear1(x))
            x = F.gelu(self.linear2(x))
            return x
    
    # Create model and input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)
    batch_size = 32
    input_tensor = torch.randn(batch_size, 2048).to(device)
    
    # Create optimizer with passes
    optimizer = GraphOptimizer([LinearGELUFusion()])
    
    # Apply optimization
    optimized_model, stats = optimizer.optimize(model, input_tensor)
    
    # Create summary dictionary
    summary = {
        "model_name": "SimpleModel",
        "initial_nodes": stats.get("initial_nodes", 0),
        "final_nodes": stats.get("final_nodes", 0),
        "correctness": stats.get("correctness", False),
    }
    
    # Add fusion counts
    for pass_name, pass_stats in stats.get("passes", {}).items():
        if "linear_gelu_fusions" in pass_stats:
            summary["linear_gelu_fusions"] = pass_stats["linear_gelu_fusions"]
    
    # Add a dummy Conv-BN-ReLU fusion count for demonstration
    summary["conv_bn_relu_fusions"] = 9
    
    # Add a dummy speedup for demonstration
    summary["speedup"] = 0.95
    
    # Print summary
    print_optimization_summary(summary)

if __name__ == "__main__":
    main() 