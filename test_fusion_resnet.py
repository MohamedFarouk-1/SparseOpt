"""Test script for Conv2d-BatchNorm2d-ReLU fusion on ResNet18."""

import torch
import torch.nn as nn
import torchvision.models as models
from rich.console import Console
import time
from sparseopt import ModelOptimizer

console = Console()

def measure_latency(model, input_tensor, num_runs=30, warmup_runs=5):
    """Measure model latency over multiple runs."""
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(input_tensor)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
    return sum(latencies) / len(latencies)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"\n[bold blue]Using device: {device}[/bold blue]")
    
    # Create model and dummy input
    model = models.resnet18(weights=None)
    model = model.to(device)
    model.eval()
    
    input_tensor = torch.randn(1, 3, 224, 224)
    input_tensor = input_tensor.to(device)
    
    # Get original output for comparison
    with torch.no_grad():
        original_output = model(input_tensor)
    
    # Measure latency before optimization
    console.print("\n[bold]Measuring baseline latency...[/bold]")
    latency_before = measure_latency(model, input_tensor)
    console.print(f"Original model latency: {latency_before:.2f} ms")
    
    # Apply optimization
    console.print("\n[bold]Applying Conv2d-BatchNorm2d-ReLU fusion...[/bold]")
    optimizer = ModelOptimizer(model, device=device)
    
    # Only apply Conv2d-BatchNorm2d-ReLU fusion patterns
    try:
        optimized_model = optimizer.optimize(fusion_patterns=['conv_bn_relu', 'conv_relu'])
    except Exception as e:
        console.print(f"[bold red]Error during optimization: {str(e)}[/bold red]")
        return
    
    # Measure latency after optimization
    console.print("\n[bold]Measuring optimized latency...[/bold]")
    latency_after = measure_latency(optimized_model, input_tensor)
    console.print(f"Optimized model latency: {latency_after:.2f} ms")
    
    # Verify outputs
    with torch.no_grad():
        optimized_output = optimized_model(input_tensor)
    
    outputs_match = torch.allclose(original_output, optimized_output, rtol=1e-3, atol=1e-3)
    console.print(f"\n[bold]{'✅' if outputs_match else '❌'} Outputs numerically close: {outputs_match}[/bold]")
    
    # Calculate and print speedup
    speedup = latency_before / latency_after
    console.print(f"\n[bold green]Speedup achieved: {speedup:.2f}x[/bold green]")
    
    # Print fusion summary
    if hasattr(optimizer, '_fused_count'):
        total_fusions = optimizer._fused_count
        if total_fusions == 0:
            console.print("\n[bold yellow]Warning: No fusions were applied. This may indicate an issue with the fusion pass.[/bold yellow]")
        else:
            console.print(f"\n[bold green]Successfully applied {total_fusions} Conv2d-BatchNorm2d-ReLU fusions[/bold green]")

if __name__ == "__main__":
    main() 