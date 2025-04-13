# SparseOpt

A toolkit for optimizing sparse and irregular PyTorch models using Torch.fx graph transformations.

## Overview

SparseOpt is a CLI/Python tool that optimizes sparse AI models (like GNNs or MoEs) by:
- Tracing them with `torch.fx`
- Applying simple graph-level optimizations (like node reordering and operator fusion)
- Benchmarking latency before and after optimization

It automatically skips models with dynamic control flow and prints a warning when optimization isn't possible.

## Features

- **Model Analysis**: Analyze PyTorch models to identify sparse operations and irregular computation patterns
- **Model Optimization**: Optimize models for better performance on sparse and irregular workloads
- **Benchmarking**: Benchmark model performance with support for both standard PyTorch models and Graph Neural Networks (GNNs)
- **Output Consistency**: Verify that optimized models produce the same output as the original
- **HuggingFace Support**: Optimize HuggingFace models with a simple API

## Installation

```bash
# Clone the repository
git clone https://github.com/MohamedFarouk-1/SparseOpt.git
cd sparseopt

# Install dependencies
pip install -r requirements.txt
```

## Quickstart

### Optimizing a HuggingFace Model

```bash
# Run the Colab test script with DistilBERT
python examples/colab_test.py --hf-model distilbert-base-uncased

# Or use the CLI
sparseopt optimize --hf-model gpt2 --output-dir results
```

### Using the Python API

```python
from sparseopt.huggingface import optimize_hf_model

# Optimize a HuggingFace model
optimized_model, stats = optimize_hf_model("gpt2")

# Print optimization statistics
print(f"Speedup: {stats['speedup']:.2f}x")
```

## Usage

### Analyzing a Model

```bash
sparseopt analyze --model model.py --class MyModel
```

### Optimizing a Model

```bash
sparseopt optimize --model model.py --class MyModel --output optimized_model.py
```

For GNN models:

```bash
sparseopt optimize --model model.py --class GNNModel --output optimized_model.py --gnn
```

### Benchmarking a Model

For standard PyTorch models:

```bash
sparseopt benchmark model.py MyModel --input-shape 1 3 224 224
```

For Graph Neural Network (GNN) models:

```bash
sparseopt benchmark gnn_model.py GNNModel --gnn --num-nodes 100 --num-edges 500 --node-features 16
```

### Testing Output Consistency

```bash
python tests/test_output_match.py --model model.py --class MyModel --input input.pt --output optimized_model.pt
```

## Example Output

```
Optimization Summary Report
==================================================
Model: SimpleGCN
✓ FX Tracing: Successful
Latency Before: 1.23 ms
Latency After: 0.87 ms
Speedup: 1.41x
Optimizations Applied:
  • Reorder
  • Fuse Linear+ReLU
==================================================
```

## Sample Models

SparseOpt includes a sample FX-compatible GCN model for testing:

- `sample_models/gcn.py`: A simple GCN model that can be traced by Torch.fx
- `sample_models/gcn.pt`: The saved model
- `sample_models/gcn_input.pt`: Sample input data for the model

To generate these files:

```bash
python examples/generate_sample_models.py
```

## Known Limitations

- **Dynamic Control Flow**: Torch.fx cannot trace models with dynamic control flow based on tensor values (e.g., `if torch.mean(x) > 0`). Such models will be saved without optimization.
- **GNN Models**: Most GNN models use PyTorch Geometric's message passing which cannot be traced by Torch.fx. These models will be saved without optimization.
- **Custom Operations**: Custom operations that cannot be traced by Torch.fx will cause optimization to fail.

## Project Structure

- `sparseopt/`: Main package directory
  - `cli.py`: Command-line interface
  - `analyze.py`: Graph analysis tools
  - `optimize.py`: Optimization passes
  - `benchmark.py`: Performance benchmarking
  - `model_loader.py`: Model loading utilities
  - `utils.py`: Shared utilities
- `tests/`: Test suite
- `sample_models/`: Sample models for testing

## Development

1. Clone the repository
2. Install development dependencies: `pip install -e ".[dev]"`
3. Run tests: `pytest tests/`

## License

MIT 