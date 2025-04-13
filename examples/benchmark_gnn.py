"""Example of benchmarking a GNN model using SparseOpt."""

import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from sparseopt.cli import benchmark
from sparseopt.benchmark import BenchmarkConfig
from sparseopt.models import ModelConfig
from sparseopt.optimizers import OptimizerConfig
from sparseopt.metrics import MetricConfig

def main():
    """Run a benchmark on a GNN model."""
    # Create model configuration
    model_config = ModelConfig(
        name="SimpleGCN",
        path="examples/simple_gnn_model.py",
        input_shape=(100, 16),  # (num_nodes, num_features)
        is_gnn=True,
        gnn_config={
            "num_nodes": 100,
            "num_edges": 500,
            "node_features": 16
        }
    )
    
    # Create optimizer configuration
    optimizer_config = OptimizerConfig(
        name="l1",
        params={
            "alpha": 0.1,
            "max_iter": 1000
        }
    )
    
    # Create metric configuration
    metric_config = MetricConfig(
        name="sparsity",
        params={
            "threshold": 1e-6
        }
    )
    
    # Create benchmark configuration
    benchmark_config = BenchmarkConfig(
        model=model_config,
        optimizer=optimizer_config,
        metrics=[metric_config],
        num_runs=5,
        output_dir="results/gnn_benchmark"
    )
    
    # Run benchmark
    benchmark(benchmark_config)

if __name__ == "__main__":
    main() 