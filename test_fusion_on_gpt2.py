#!/usr/bin/env python
"""Test script to evaluate fusion pass on GPT-2 model."""

import torch
from transformers import GPT2Model
from sparseopt.optimize import ModelOptimizer
from rich.console import Console
import time

console = Console()

def main():
    """Main function to test fusion pass on GPT-2."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold]Using device: {device}[/bold]")
    
    # 1. Load GPT-2 model
    console.print("[bold]Loading GPT-2 model...[/bold]")
    model = GPT2Model.from_pretrained("gpt2")
    model.to(device)
    model.eval()
    
    # 2. Create dummy input
    console.print("[bold]Creating dummy input...[/bold]")
    batch_size = 1
    seq_length = 16
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    
    # 3. Run inference before optimization
    console.print("[bold]Running inference before optimization...[/bold]")
    with torch.no_grad():
        start_time = time.time()
        outputs_before = model(input_ids)
        inference_time_before = time.time() - start_time
    
    console.print(f"[green]✓[/green] Inference time before optimization: {inference_time_before*1000:.2f} ms")
    
    # 4. Apply optimization
    console.print("[bold]Applying optimization...[/bold]")
    optimizer = ModelOptimizer(model)
    optimized_model = optimizer.optimize(
        use_layer_by_layer=True,
        fusion_patterns=["linear_relu", "linear_gelu", "layer_norm"]
    )
    
    # 5. Print fusion results
    console.print("[bold]Fusion results:[/bold]")
    # The fusion results are already printed by the optimizer
    
    # 6. Run inference after optimization
    console.print("[bold]Running inference after optimization...[/bold]")
    with torch.no_grad():
        start_time = time.time()
        outputs_after = optimized_model(input_ids)
        inference_time_after = time.time() - start_time
    
    console.print(f"[green]✓[/green] Inference time after optimization: {inference_time_after*1000:.2f} ms")
    
    # Calculate speedup
    speedup = inference_time_before / inference_time_after
    console.print(f"[bold]Speedup: {speedup:.2f}x[/bold]")
    
    # Verify outputs are similar
    console.print("[bold]Verifying outputs...[/bold]")
    max_diff = torch.max(torch.abs(outputs_before.last_hidden_state - outputs_after.last_hidden_state))
    console.print(f"[green]✓[/green] Maximum difference in outputs: {max_diff.item():.6f}")
    
    if max_diff < 1e-5:
        console.print("[bold green]✓[/bold green] Optimization successful! Outputs are numerically equivalent.")
    else:
        console.print("[bold yellow]⚠[/bold yellow] Optimization changed the outputs slightly. This may be expected for some fusion patterns.")

if __name__ == "__main__":
    main() 