import unittest
import torch
import torchvision
from rich.console import Console
from rich.table import Table
import time
from sparseopt.graph.optimizer import GraphOptimizer
from sparseopt.graph.fusion import ConvBnReluGraphFusion

class TestResNetFusion(unittest.TestCase):
    def setUp(self):
        # Create models
        self.model = torchvision.models.resnet18(weights='DEFAULT')
        self.model.eval()
        
        # Create test input
        self.input_tensor = torch.randn(1, 3, 224, 224)
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.input_tensor = self.input_tensor.to(self.device)
        
        # Create optimizer with fusion pass
        self.optimizer = GraphOptimizer([ConvBnReluGraphFusion()])
        
        # Console for pretty printing
        self.console = Console()
        
        # Print device info
        self.console.print(f"\n[bold blue]Running tests on: {self.device}[/bold blue]")
    
    def measure_latency(self, model, num_runs=100, warmup=10):
        """Measure model latency."""
        # Warmup runs
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(self.input_tensor)
        
        # Measure latency
        latencies = []
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(self.input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)
        
        return sum(latencies) / len(latencies) * 1000  # Convert to ms
    
    def print_graph(self, graph_module, title):
        """Print the FX graph structure."""
        self.console.print(f"\n[bold cyan]{title}:[/bold cyan]")
        for node in graph_module.graph.nodes:
            self.console.print(f"Node: {node.op} - Target: {node.target}")
            if hasattr(node, 'args'):
                self.console.print(f"  Args: {node.args}")
            if hasattr(node, 'kwargs'):
                self.console.print(f"  Kwargs: {node.kwargs}")
    
    def test_fusion_correctness(self):
        """Test that fusion maintains numerical correctness."""
        # Trace the model
        traced_model = torch.fx.symbolic_trace(self.model)
        self.print_graph(traced_model, "Original Graph")
        
        # Get original output
        with torch.no_grad():
            original_output = self.model(self.input_tensor)
        
        # Apply fusion
        optimized_model, stats = self.optimizer.optimize(
            self.model,
            test_input=self.input_tensor
        )
        
        # Print optimized graph
        traced_optimized = torch.fx.symbolic_trace(optimized_model)
        self.print_graph(traced_optimized, "Optimized Graph")
        
        # Get optimized output
        with torch.no_grad():
            optimized_output = optimized_model(self.input_tensor)
        
        # Verify outputs match
        self.assertTrue(
            torch.allclose(original_output, optimized_output, rtol=1e-3, atol=1e-3),
            "Fusion changed model output"
        )
        
        # Print fusion statistics
        self.console.print("\n[bold green]Fusion Statistics:[/bold green]")
        stats_table = Table(show_header=True, header_style="bold magenta")
        stats_table.add_column("Metric")
        stats_table.add_column("Value")
        
        stats_table.add_row("Fused Operations", str(stats['pass_0']['fused_count']))
        stats_table.add_row("Fusion Details", str(len(stats['pass_0']['fusion_stats'])))
        
        self.console.print(stats_table)
    
    def test_fusion_performance(self):
        """Test that fusion improves performance."""
        # Measure original latency
        original_latency = self.measure_latency(self.model)
        
        # Apply fusion
        optimized_model, _ = self.optimizer.optimize(
            self.model,
            test_input=self.input_tensor
        )
        
        # Measure optimized latency
        optimized_latency = self.measure_latency(optimized_model)
        
        # Print performance results
        self.console.print("\n[bold green]Performance Results:[/bold green]")
        perf_table = Table(show_header=True, header_style="bold magenta")
        perf_table.add_column("Model")
        perf_table.add_column("Latency (ms)")
        perf_table.add_column("Speedup")
        
        perf_table.add_row(
            "Original",
            f"{original_latency:.2f}",
            "1.00x"
        )
        perf_table.add_row(
            "Optimized",
            f"{optimized_latency:.2f}",
            f"{original_latency/optimized_latency:.2f}x"
        )
        
        self.console.print(perf_table)
        
        # Verify performance improvement
        self.assertLess(
            optimized_latency,
            original_latency,
            "Fusion did not improve performance"
        )

if __name__ == '__main__':
    unittest.main() 