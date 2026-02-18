#!/usr/bin/env python3
"""
SparseOpt CLI — optimize any supported PyTorch model with one command.

Usage:
    python sparseopt.py --model resnet50 --device cpu
    python sparseopt.py --model bert-base --device cpu
    python sparseopt.py --model mlp --device cpu
    python sparseopt.py --model resnet50 --device cpu --runs 200 --warmup 20
    python sparseopt.py --model resnet50 --passes dropout,dead_node,conv_fusion
    python sparseopt.py --model resnet50 --save-results benchmarks/results.json
"""

import argparse
import json
import sys
import time

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sparseopt.models import get_demo_model
from sparseopt.core import optimize_model, DEFAULT_PASSES, PASS_REGISTRY
from sparseopt.benchmark import measure_latency_and_memory

console = Console()


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SparseOpt — PyTorch FX computation graph optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["resnet50", "bert-base", "mlp"],
        help="Demo model to optimize",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on (default: cpu)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of timed benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations before timing (default: 10)",
    )
    parser.add_argument(
        "--passes",
        default=None,
        help=(
            "Comma-separated list of passes to apply. "
            f"Default: all ({', '.join(DEFAULT_PASSES)}). "
            f"Available: {', '.join(PASS_REGISTRY)}"
        ),
    )
    parser.add_argument(
        "--save-results",
        default=None,
        metavar="PATH",
        help="Write JSON results to this path",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip numerical correctness verification (faster)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _pct(before: float, after: float) -> str:
    if before == 0:
        return "N/A"
    pct = (after - before) / before * 100
    arrow = "↓" if pct < 0 else "↑"
    return f"{pct:+.1f}% {arrow}"


def _speedup(before: float, after: float) -> str:
    if after == 0:
        return "N/A"
    return f"{before / after:.2f}×"


def print_results_table(
    model_name: str,
    device: str,
    before: dict,
    after: dict,
    opt_stats: dict,
) -> None:
    method = opt_stats.get("method", "unknown")
    node_before = opt_stats.get("node_count_before", "N/A")
    node_after = opt_stats.get("node_count_after", "N/A")
    fusions = opt_stats.get("total_fusions", 0)
    correct = opt_stats.get("correct", None)

    table = Table(
        title=f"[bold cyan]SparseOpt Results — {model_name} ({device})[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
        min_width=72,
    )
    table.add_column("Metric", style="cyan", min_width=28)
    table.add_column("Before", justify="right", min_width=14)
    table.add_column("After", justify="right", min_width=14)
    table.add_column("Change", justify="right", min_width=14)

    lat_b = before["latency_ms"]
    lat_a = after["latency_ms"]
    std_b = before["latency_std_ms"]
    std_a = after["latency_std_ms"]
    mem_b = before["peak_memory_mb"]
    mem_a = after["peak_memory_mb"]

    table.add_row(
        "Latency — mean (ms)",
        f"{lat_b:.2f}",
        f"{lat_a:.2f}",
        f"{_pct(lat_b, lat_a)}  {_speedup(lat_b, lat_a)}",
    )
    table.add_row(
        "Latency — std (ms)",
        f"{std_b:.2f}",
        f"{std_a:.2f}",
        "",
    )
    if mem_b > 0 or mem_a > 0:
        table.add_row(
            "Peak Memory (MB)",
            f"{mem_b:.1f}",
            f"{mem_a:.1f}",
            _pct(mem_b, mem_a) if mem_b > 0 else "N/A",
        )
    if node_before != "N/A":
        nodes_elim = (
            f"-{node_before - node_after}"
            if isinstance(node_before, int) and isinstance(node_after, int)
            else "N/A"
        )
        table.add_row(
            "Graph Nodes",
            str(node_before),
            str(node_after),
            nodes_elim,
        )
    table.add_row(
        "Operators Fused",
        "—",
        str(fusions),
        f"+{fusions}" if fusions else "—",
    )
    if correct is not None:
        correctness_str = "[green]✓ pass[/green]" if correct else "[red]✗ fail[/red]"
        table.add_row(
            "Numerical Correctness (1e-4)",
            "—",
            correctness_str,
            "",
        )
    table.add_row(
        "Optimization Strategy",
        "—",
        method.replace("_", " "),
        "",
    )

    console.print()
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    passes = (
        [p.strip() for p in args.passes.split(",")]
        if args.passes
        else None
    )

    # ── Header ───────────────────────────────────────────────────────────
    console.print(
        Panel.fit(
            f"[bold]SparseOpt[/bold]  ·  PyTorch FX Graph Optimizer\n"
            f"Model: [cyan]{args.model}[/cyan]   "
            f"Device: [cyan]{args.device}[/cyan]   "
            f"Runs: [cyan]{args.runs}[/cyan]   "
            f"Warmup: [cyan]{args.warmup}[/cyan]",
            border_style="cyan",
        )
    )

    # ── Load model ───────────────────────────────────────────────────────
    with console.status(f"Loading [cyan]{args.model}[/cyan]…"):
        try:
            model, example_inputs = get_demo_model(args.model, device=args.device)
        except Exception as e:
            console.print(f"[red]Failed to load model:[/red] {e}")
            sys.exit(1)

    param_count = sum(p.numel() for p in model.parameters())
    console.print(
        f"  [green]✓[/green] Loaded {args.model}  "
        f"({param_count / 1e6:.1f}M parameters)"
    )

    # ── Baseline benchmark ───────────────────────────────────────────────
    with console.status("Benchmarking [dim]baseline[/dim]…"):
        before = measure_latency_and_memory(
            model, example_inputs, device=args.device,
            num_runs=args.runs, warmup=args.warmup,
        )
    console.print(
        f"  [green]✓[/green] Baseline: "
        f"[yellow]{before['latency_ms']:.2f} ms[/yellow] avg"
    )

    # ── Optimize ─────────────────────────────────────────────────────────
    with console.status("Optimizing…"):
        t0 = time.perf_counter()
        try:
            optimized, opt_stats = optimize_model(
                model,
                example_inputs,
                passes=passes,
                verify=not args.no_verify,
            )
        except Exception as e:
            console.print(f"[red]Optimization failed:[/red] {e}")
            sys.exit(1)
    opt_time = (time.perf_counter() - t0) * 1000
    console.print(
        f"  [green]✓[/green] Optimized in {opt_time:.0f} ms  "
        f"([cyan]{opt_stats.get('total_fusions', 0)}[/cyan] fusions, "
        f"strategy: {opt_stats.get('method', '?')})"
    )

    # ── Post-optimization benchmark ───────────────────────────────────────
    with console.status("Benchmarking [dim]optimized model[/dim]…"):
        after = measure_latency_and_memory(
            optimized, example_inputs, device=args.device,
            num_runs=args.runs, warmup=args.warmup,
        )
    console.print(
        f"  [green]✓[/green] Optimized: "
        f"[yellow]{after['latency_ms']:.2f} ms[/yellow] avg"
    )

    # ── Results table ────────────────────────────────────────────────────
    print_results_table(args.model, args.device, before, after, opt_stats)

    # ── Save JSON (optional) ─────────────────────────────────────────────
    if args.save_results:
        import os, pathlib
        pathlib.Path(args.save_results).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": args.model,
            "device": args.device,
            "runs": args.runs,
            "warmup": args.warmup,
            "before": before,
            "after": after,
            "optimization_stats": {
                k: v for k, v in opt_stats.items()
                if isinstance(v, (int, float, bool, str))
            },
        }
        with open(args.save_results, "w") as f:
            json.dump(payload, f, indent=2)
        console.print(f"  [green]✓[/green] Results saved to [cyan]{args.save_results}[/cyan]")


if __name__ == "__main__":
    main()
