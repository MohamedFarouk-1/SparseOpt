# SparseOpt

**A PyTorch computation graph optimizer that automatically fuses operators, eliminates dead nodes, and reorders execution — with zero changes to your model code.**

Most ML engineers treat model inference as a black box. SparseOpt opens the graph, applies a compiler-style optimization pipeline via `torch.fx`, verifies numerical correctness to `1e-4` tolerance, and hands back a faster model.

---

## The Problem

PyTorch models are written for readability, not execution efficiency. A typical forward pass is full of redundant structure: activations that could be folded into the preceding linear layer, dead nodes that exist only because of how the model was assembled, dropout ops that serve no purpose at inference time, and operator sequences that stall the hardware pipeline. Most developers don't know how to touch the computation graph directly — and shouldn't have to.

SparseOpt automates the graph-level optimizations that compiler engineers apply by hand.

---

## How It Works

SparseOpt uses `torch.fx` to trace a model into a symbolic computation graph, then runs a pipeline of optimization passes over that graph:

```
nn.Module
    │
    ▼
torch.fx.symbolic_trace()          ← captures the full computation graph
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Optimization Pipeline                                   │
│                                                         │
│  1. DropoutElimination      remove no-op inference nodes│
│  2. DeadNodeElimination     prune unreachable subgraphs  │
│  3. NodeReordering          schedule lightweight ops first│
│  4. Operator Fusion         collapse sequential patterns │
│     ├─ Conv2d + BN + ReLU → ConvBnReLU2d               │
│     ├─ Conv2d + ReLU      → ConvReLU2d                  │
│     ├─ Linear + ReLU      → FusedLinearReLU             │
│     ├─ Linear + GELU      → FusedLinearGELU             │
│     └─ LayerNorm + Linear + LayerNorm → fused block     │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Numerical Verification             ← torch.allclose(rtol=1e-4, atol=1e-4)
    │
    ▼
Benchmarking                       ← 10 warmup + 100 timed iterations
    │
    ▼
Optimized nn.Module
```

For models with dynamic control flow (LLMs, MoE architectures) that resist whole-graph tracing, SparseOpt falls back to a layer-by-layer tracing strategy, applying passes to each traceable submodule independently.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Graph IR | `torch.fx` — PyTorch's built-in symbolic tracer |
| Fusion primitives | `torch.nn.intrinsic` (`ConvBnReLU2d`, `ConvReLU2d`) |
| Model loading | HuggingFace `transformers` + `AutoModel` |
| Verification | `torch.allclose` with configurable tolerances |
| Output formatting | `rich` tables and console |
| Python | 3.8+ |
| PyTorch | ≥ 1.10.0 |

---

## Installation

```bash
git clone https://github.com/MohamedFarouk-1/SparseOpt.git
cd SparseOpt
pip install -r requirements.txt
pip install -e .
```

---

## Usage

### CLI — optimize any HuggingFace model in one command

```bash
python optimize.py --model bert-base-uncased --device cuda
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | HuggingFace model name or local path |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--num-runs` | `10` | Benchmark iterations |
| `--warmup` | `3` | Warmup runs before timing |
| `--text` | `"Hello, this is SparseOpt!"` | Input text |
| `--max-length` | `128` | Tokenizer max sequence length |

---

### Python API

```python
import torch
from sparseopt.huggingface import load_hf_model, create_hf_input, optimize_hf_model, benchmark_model

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model, tokenizer = load_hf_model("bert-base-uncased", device=device)
inputs = create_hf_input(tokenizer, "Inference optimization with SparseOpt", device=device)

# Baseline
baseline = benchmark_model(model, inputs, device=device, num_runs=50, warmup=10)
print(f"Baseline:  {baseline['mean_latency']:.2f} ms")

# Optimize
optimized_model, stats = optimize_hf_model("bert-base-uncased", device=device)

# Optimized
result = benchmark_model(optimized_model, inputs, device=device, num_runs=50, warmup=10)
print(f"Optimized: {result['mean_latency']:.2f} ms")
print(f"Speedup:   {baseline['mean_latency'] / result['mean_latency']:.2f}x")
```

### Compose passes manually

```python
import torch
import torch.fx as fx
from sparseopt.graph.passes.dead_node import DeadNodeEliminationPass
from sparseopt.graph.passes.reordering import NodeReorderingPass
from sparseopt.graph.passes.linear_fusion import LinearGELUFusion
from sparseopt.graph.optimizer_pipeline import OptimizerPipeline

model = ...   # any nn.Module
inputs = {"input_ids": torch.randint(0, 1000, (1, 64))}

gm = fx.symbolic_trace(model)

pipeline = OptimizerPipeline(
    passes=[
        DeadNodeEliminationPass(),
        NodeReorderingPass(),
        LinearGELUFusion(),
    ],
    verify_correctness=True,
    benchmark=True,
)

optimized_gm, stats = pipeline.optimize(gm, example_inputs=inputs)
```

---

## Benchmark Results

Measured on CPU (100 runs, 10 warmup iterations). Run with `python benchmarks/run_benchmarks.py`.

| Model | Params | Baseline (ms) | SparseOpt (ms) | Speedup | Optimizations Applied |
|-------|--------|:-------------:|:--------------:|:-------:|----------------------|
| **MLP** | 6.3M | 0.33 | 0.33 | **1.01×** | Dropout elimination ×2, Linear+ReLU fusion ×3, 11→6 graph nodes |
| **ResNet-50** | 25.6M | 14.90 | 14.59 | **1.02×** | Node reordering (layer-by-layer) |
| **BERT-base** | 109.5M | 18.35 | 17.58 | **1.04×** | Linear+GELU fusion ×36 across 37 traced submodules |

> CPU gains are conservative; GPU workloads see larger improvements because fused kernels reduce kernel-launch overhead. BERT's 4.2% reduction compounds across all 12 transformer layers — the same fusion pass fires 36 times. MLP graph compression (11 → 6 nodes, correctness verified to 1e-4) lowers memory bandwidth at higher batch sizes.

---

## Supported Fusion Patterns

| Pattern | Target Architecture |
|---------|-------------|
| `Conv2d → BatchNorm → ReLU` | CNNs (ResNet, EfficientNet, ...) |
| `Conv2d → ReLU` | CNNs |
| `Linear → ReLU` | MLPs, classifier heads |
| `Linear → GELU` | Transformers (GPT, BERT, ...) |
| `LayerNorm → Linear → LayerNorm` | Transformer blocks |

---

## Tested Models

- `bert-base-uncased`
- `distilbert-base-uncased`
- `facebook/opt-350m`
- `gpt2`
- ResNet-18 / ResNet-50 (torchvision)
- Custom MLP and GNN models

---

## Project Structure

```
sparseopt/
├── graph/
│   ├── passes/
│   │   ├── dead_node.py           # Dead node elimination
│   │   ├── reordering.py          # Node reordering scheduler
│   │   ├── dropout_elimination.py # Inference dropout removal
│   │   ├── linear_fusion.py       # Linear+GELU / Linear+ReLU fusion
│   │   └── conv_fusion.py         # Conv+BN+ReLU / Conv+ReLU fusion
│   ├── optimizer_pipeline.py      # Pipeline: apply → verify → benchmark
│   └── base.py                    # GraphPass base class
├── huggingface.py                 # HuggingFace model helpers
├── optimize.py                    # Core ModelOptimizer class
└── utils/
    └── benchmark.py               # Timing utilities
examples/
├── huggingface_optimization.py    # Full HuggingFace example
├── optimize_llm.py                # LLM layer-by-layer tracing
└── benchmark_gnn.py               # GNN benchmarking
```

---

## License

MIT
