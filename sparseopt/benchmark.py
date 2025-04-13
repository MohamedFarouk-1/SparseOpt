"""Benchmarking utilities for SparseOpt."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, Union
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import inspect

from .errors import (
    TorchFxTracingError,
    InputShapeMismatchError,
    validate_input_shape
)

# Try to import PyTorch Geometric
try:
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    Data = None

class ModelBenchmarker:
    """Class for benchmarking PyTorch models."""
    
    def __init__(self, model, input_data, device='cpu'):
        """Initialize the benchmarker.
        
        Args:
            model: PyTorch model to benchmark
            input_data: Input data for the model
            device: Device to run benchmarks on (default: "cpu")
        """
        self.model = model.to(device)
        self.input_data = input_data.to(device)  # ✅ make sure this is defined
        self.device = device

    def measure_latency(self, warmup=5, runs=20):
        """Measure model latency with warmup runs.
        
        Args:
            warmup: Number of warmup runs (default: 5)
            runs: Number of timed runs (default: 20)
            
        Returns:
            Average latency in seconds
        """
        self.model.eval()
        with torch.no_grad():
            # Warm-up
            for _ in range(warmup):
                _ = self.model(self.input_data)

            # Timed runs
            timings = []
            for _ in range(runs):
                start = time.time()
                _ = self.model(self.input_data)
                end = time.time()
                timings.append(end - start)

        avg_latency = sum(timings) / len(timings)
        print(f"✅ Average Latency: {avg_latency * 1000:.2f} ms")
        return avg_latency

    def _is_gnn_model(self) -> bool:
        """Check if the model is a GNN model that expects a Data object.
        
        Returns:
            True if the model is a GNN model, False otherwise
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            return False
            
        # Get the forward method signature
        forward_sig = inspect.signature(self.model.forward)
        
        # Check if the first parameter is annotated as Data
        if len(forward_sig.parameters) > 0:
            first_param = list(forward_sig.parameters.values())[0]
            if first_param.annotation is not inspect.Parameter.empty:
                # Safely check if the annotation is Data from torch_geometric
                return first_param.annotation.__name__ == 'Data' and 'torch_geometric' in first_param.annotation.__module__
                
        return False
        
    def _create_dummy_input(self) -> Union[torch.Tensor, Data]:
        """Create a dummy input for the model.
        
        Returns:
            A dummy input tensor or Data object
        """
        if self.gnn:
            # Create node features
            x = torch.randn((self.num_nodes, self.node_features), device=self.device)
            
            # Create random edge index
            edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges), device=self.device)
            
            # Create Data object
            return Data(x=x, edge_index=edge_index)
        else:
            # Create regular tensor input
            if self.input_shape is None:
                raise ValueError("Input shape is required for non-GNN models")
            return torch.randn(self.input_shape, device=self.device)
        
    def run_benchmark(self) -> Dict[str, float]:
        """Run benchmarks on the model.
        
        Returns:
            Dictionary containing benchmark metrics
            
        Raises:
            TorchFxTracingError: If model tracing fails
            InputShapeMismatchError: If input shape doesn't match model's expected shape
        """
        try:
            # Create dummy input
            dummy_input = self._create_dummy_input()
            
            # Warmup runs
            for _ in range(self.warmup_runs):
                with torch.no_grad():
                    _ = self.model(dummy_input)
                    
            # Benchmark runs
            latencies = []
            for _ in range(self.num_runs):
                start_time = time.perf_counter()
                with torch.no_grad():
                    _ = self.model(dummy_input)
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
                
            # Calculate statistics
            avg_latency = sum(latencies) / len(latencies)
            p50_latency = sorted(latencies)[len(latencies) // 2]
            p90_latency = sorted(latencies)[int(len(latencies) * 0.9)]
            p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
            
            return {
                "avg_latency": avg_latency,
                "p50_latency": p50_latency,
                "p90_latency": p90_latency,
                "p99_latency": p99_latency,
                "throughput": 1000 / avg_latency  # Samples per second
            }
            
        except RuntimeError as e:
            if "Expected input" in str(e) and "to have" in str(e):
                raise InputShapeMismatchError(str(e))
            raise TorchFxTracingError(str(e))
            
    def print_benchmark_results(self, results: Dict[str, float]) -> None:
        """Print benchmark results in a formatted table.
        
        Args:
            results: Dictionary containing benchmark metrics
        """
        table = Table(title="Benchmark Results")
        
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Model", self.model.__class__.__name__)
        table.add_row("Device", self.device)
        
        if self.gnn:
            dummy_input = self._create_dummy_input()
            table.add_row("Model Type", "GNN (PyTorch Geometric)")
            table.add_row("Node Features Shape", str(dummy_input.x.shape))
            table.add_row("Edge Index Shape", str(dummy_input.edge_index.shape))
        else:
            table.add_row("Model Type", "Standard PyTorch")
            table.add_row("Input Shape", str(self.input_shape))
            
        table.add_row("Average Latency", f"{results['avg_latency']:.2f} ms")
        table.add_row("P50 Latency", f"{results['p50_latency']:.2f} ms")
        table.add_row("P90 Latency", f"{results['p90_latency']:.2f} ms")
        table.add_row("P99 Latency", f"{results['p99_latency']:.2f} ms")
        table.add_row("Throughput", f"{results['throughput']:.2f} samples/s")
        
        self.console.print(table)
        
    def compare_models(
        self,
        other_model: nn.Module,
        other_input_shape: Optional[Tuple[int, ...]] = None,
        other_gnn: Optional[bool] = None,
        other_num_nodes: Optional[int] = None,
        other_num_edges: Optional[int] = None,
        other_node_features: Optional[int] = None
    ) -> None:
        """Compare benchmark results between two models.
        
        Args:
            other_model: Second model to benchmark
            other_input_shape: Input shape for second model (default: same as self)
            other_gnn: Whether the second model is a GNN model (default: same as self)
            other_num_nodes: Number of nodes for second GNN model (default: same as self)
            other_num_edges: Number of edges for second GNN model (default: same as self)
            other_node_features: Number of node features for second GNN model (default: same as self)
            
        Raises:
            TorchFxTracingError: If model tracing fails
            InputShapeMismatchError: If input shape doesn't match model's expected shape
        """
        other_input_shape = other_input_shape or self.input_shape
        other_gnn = other_gnn if other_gnn is not None else self.gnn
        other_num_nodes = other_num_nodes or self.num_nodes
        other_num_edges = other_num_edges or self.num_edges
        other_node_features = other_node_features or self.node_features
        
        # Create second benchmarker
        other_benchmarker = ModelBenchmarker(
            other_model,
            input_shape=other_input_shape,
            device=self.device,
            num_runs=self.num_runs,
            warmup_runs=self.warmup_runs,
            gnn=other_gnn,
            num_nodes=other_num_nodes,
            num_edges=other_num_edges,
            node_features=other_node_features
        )
        
        # Run benchmarks
        results1 = self.run_benchmark()
        results2 = other_benchmarker.run_benchmark()
        
        # Print comparison
        table = Table(title="Model Comparison")
        
        table.add_column("Metric", style="cyan")
        table.add_column(self.model.__class__.__name__, style="green")
        table.add_column(other_model.__class__.__name__, style="blue")
        table.add_column("Difference", style="yellow")
        
        metrics = [
            ("Average Latency", "avg_latency", "ms"),
            ("P50 Latency", "p50_latency", "ms"),
            ("P90 Latency", "p90_latency", "ms"),
            ("P99 Latency", "p99_latency", "ms"),
            ("Throughput", "throughput", "samples/s")
        ]
        
        for name, key, unit in metrics:
            v1 = results1[key]
            v2 = results2[key]
            diff = ((v2 - v1) / v1) * 100
            
            table.add_row(
                name,
                f"{v1:.2f} {unit}",
                f"{v2:.2f} {unit}",
                f"{diff:+.2f}%"
            )
            
        self.console.print(table) 