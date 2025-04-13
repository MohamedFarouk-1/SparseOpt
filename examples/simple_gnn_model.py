"""Example of a simple Graph Convolutional Network (GCN) model."""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GCNModel(torch.nn.Module):
    """A simple Graph Convolutional Network model.
    
    This model consists of two GCN layers followed by a linear layer for node classification.
    """
    
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=2):
        """Initialize the GCN model.
        
        Args:
            input_dim: Number of input node features
            hidden_dim: Number of hidden channels in the GCN layers
            output_dim: Number of output classes for node classification
        """
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node features matrix
                - edge_index: Graph connectivity in COO format
                
        Returns:
            Node classification logits
        """
        x, edge_index = data.x, data.edge_index
        
        # First GCN layer
        x = F.relu(self.conv1(x, edge_index))
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        
        return x

def create_random_graph(num_nodes: int = 100, num_edges: int = 500, num_features: int = 16) -> Data:
    """Create a random graph for testing.
    
    Args:
        num_nodes: Number of nodes in the graph
        num_edges: Number of edges in the graph
        num_features: Number of features per node
        
    Returns:
        PyTorch Geometric Data object containing a random graph
    """
    # Create random node features
    x = torch.randn(num_nodes, num_features)
    
    # Create random edges (undirected graph)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Make edges undirected
    
    # Create random node labels
    y = torch.randint(0, 3, (num_nodes,))  # 3 classes
    
    return Data(x=x, edge_index=edge_index, y=y)

if __name__ == "__main__":
    # Create a random graph
    data = create_random_graph()
    
    # Create and test the model
    model = GCNModel(input_dim=16, hidden_dim=32, output_dim=3)
    out = model(data)
    
    print(f"Input graph: {data.num_nodes} nodes, {data.num_edges // 2} edges")
    print(f"Model output shape: {out.shape}")
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}") 