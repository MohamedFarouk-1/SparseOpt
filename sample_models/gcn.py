"""Sample FX-compatible GCN model for SparseOpt testing."""

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import os
from pathlib import Path

# Define the model class at the top level so it's importable
class SimpleGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(16, 32)
        self.conv2 = GCNConv(32, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def create_sample_graph(num_nodes=5, num_edges=8, num_features=16):
    """Create a sample graph for testing.
    
    Args:
        num_nodes: Number of nodes in the graph
        num_edges: Number of edges in the graph
        num_features: Number of features per node
        
    Returns:
        PyTorch Geometric Data object
    """
    # Create random node features
    x = torch.randn(num_nodes, num_features)
    
    # Create random edges (ensure they're valid)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Create a Data object
    data = Data(x=x, edge_index=edge_index)
    
    return data

def save_sample_model_and_input():
    """Save the sample model and input for testing."""
    # Create output directory if it doesn't exist
    output_dir = Path("sample_models")
    output_dir.mkdir(exist_ok=True)
    
    # Create model and input
    model = SimpleGCN()
    data = create_sample_graph()
    
    # Save model
    torch.save(model, output_dir / "gcn.pt")
    
    # Save input
    torch.save(data, output_dir / "gcn_input.pt")
    
    print(f"Saved model to {output_dir / 'gcn.pt'}")
    print(f"Saved input to {output_dir / 'gcn_input.pt'}")
    
    # Print model information
    print("\nModel Information:")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Input shape: {data.x.shape}")
    print(f"Number of edges: {data.edge_index.shape[1]}")
    
    return model, data

# Make the class importable
__all__ = ["SimpleGCN"]

# Only run this block when executing directly
if __name__ == "__main__":
    # Create and save the model and input data
    model, data = save_sample_model_and_input()
    
    # Save the model state_dict and input data for SparseOpt in the current directory
    print("\nSaving model state_dict and input data for SparseOpt in the current directory...")
    torch.save(model.state_dict(), "gcn.pt")
    torch.save(data, "gcn_input.pt")
    print("✅ Saved model to gcn.pt")
    print("✅ Saved input to gcn_input.pt") 