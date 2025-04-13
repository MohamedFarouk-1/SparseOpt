import torch
import torch.nn as nn
import torch.nn.functional as F
from sparseopt.graph.optimizer import GraphOptimizer
from sparseopt.graph.passes import LinearGELUFusion
from rich.console import Console
from rich.table import Table
import time

def measure_latency(model, input_tensor, num_runs=100, warmup=50):
    """Measure model latency over multiple runs."""
    # Warmup runs
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_tensor)
    
    # Actual measurement
    latencies = []
    for _ in range(num_runs):
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(input_tensor)
                torch.cuda.synchronize()
                end = time.perf_counter()
            else:
                start = time.perf_counter()
                _ = model(input_tensor)
                end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to milliseconds
    
    return sum(latencies) / len(latencies)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 2048  # Increased from 512
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = F.gelu(self.linear2(x))
        return x

def main():
    # Create model and move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)
    
    # Create input tensor with larger batch size
    batch_size = 32  # Added batch dimension
    input_tensor = torch.randn(batch_size, 2048).to(device)
    
    # Measure original model latency
    original_latency = measure_latency(model, input_tensor)
    
    # Apply optimization
    optimizer = GraphOptimizer([LinearGELUFusion()])
    optimized_model, stats = optimizer.optimize(model, input_tensor)
    
    # Measure optimized model latency
    optimized_latency = measure_latency(optimized_model, input_tensor)
    
    # Verify correctness
    with torch.no_grad():
        original_output = model(input_tensor)
        optimized_output = optimized_model(input_tensor)
        is_correct = torch.allclose(original_output, optimized_output, rtol=1e-3, atol=1e-3)
    
    # Create results table
    console = Console()
    table = Table(title="Linear-GELU Fusion Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Number of Fusions", str(stats.get("num_fusions", 0)))
    table.add_row("Original Latency (ms)", f"{original_latency:.2f}")
    table.add_row("Optimized Latency (ms)", f"{optimized_latency:.2f}")
    table.add_row("Speedup", f"{original_latency/optimized_latency:.2f}x")
    table.add_row("Correctness", "✓" if is_correct else "✗")
    
    console.print(table)
    
    # Print fusion details if any fusions were applied
    if stats.get("num_fusions", 0) > 0:
        console.print("\nFusion Details:")
        for i, fusion in enumerate(stats.get("fusions", []), 1):
            console.print(f"Fusion {i}: {fusion}")

if __name__ == "__main__":
    main() 