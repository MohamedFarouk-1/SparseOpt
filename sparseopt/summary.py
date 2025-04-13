import torch
from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel

def print_optimization_summary(summary_dict: Dict[str, Any]) -> None:
    """
    Print a formatted optimization summary to the terminal.
    
    Args:
        summary_dict: Dictionary containing optimization statistics
    """
    console = Console()
    
    # Create a table for the summary
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Add model name if available
    if "model_name" in summary_dict:
        table.add_row("Model", summary_dict["model_name"])
    
    # Add fusion counts
    if "conv_bn_relu_fusions" in summary_dict:
        table.add_row("• Conv-BN-ReLU fused", str(summary_dict["conv_bn_relu_fusions"]))
    
    if "linear_gelu_fusions" in summary_dict:
        table.add_row("• Linear-GELU fused", str(summary_dict["linear_gelu_fusions"]))
    
    # Add graph node counts
    if "initial_nodes" in summary_dict and "final_nodes" in summary_dict:
        table.add_row(
            "• Graph nodes reduced", 
            f"{summary_dict['initial_nodes']} → {summary_dict['final_nodes']}"
        )
    
    # Add correctness status
    if "correctness" in summary_dict:
        correctness_text = Text("PASSED", style="green") if summary_dict["correctness"] else Text("FAILED", style="red")
        table.add_row("• Numerical correctness", correctness_text)
    
    # Add speedup if available
    if "speedup" in summary_dict:
        table.add_row("• CPU speedup", f"{summary_dict['speedup']:.2f}x")
    
    # Print the table in a panel
    console.print(Panel(table, title="✅ Optimization Summary", border_style="green")) 