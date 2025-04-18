# SparseOpt

SparseOpt is an AI model graph optimizer built on PyTorch FX. It provides tools for optimizing and benchmarking transformer models from HuggingFace.

## Features

- **Graph Optimization**: Fuses common patterns in transformer models for improved performance
- **Benchmarking**: Measures and compares performance before and after optimization
- **Numerical Correctness**: Validates that optimized models produce the same outputs as the original
- **HuggingFace Integration**: Seamlessly works with models from the HuggingFace Hub

## Installation

```bash
# Clone the repository
git clone https://github.com/MohamedFarouk-1/SparseOpt.git
cd SparseOpt

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Usage

### Command Line Interface

The simplest way to use SparseOpt is through the command-line interface:

```bash
python optimize.py --model bert-base-uncased --device cuda
```

This will:
1. Load the specified model from HuggingFace
2. Benchmark the original model
3. Apply optimization passes
4. Benchmark the optimized model
5. Validate numerical correctness
6. Print a summary of the results

#### Command-line Arguments

- `--model`: Name of the HuggingFace model (required)
- `--device`: Device to run on (`cuda` or `cpu`, default: `cuda`)
- `--num-runs`: Number of benchmark runs (default: 10)
- `--warmup`: Number of warmup runs (default: 3)
- `--text`: Input text for benchmarking (default: "Hello, this is SparseOpt!")
- `--max-length`: Maximum sequence length (default: 128)

### Google Colab

You can also use SparseOpt in Google Colab. Here's a simple example:

```python
# Install dependencies
!pip install torch torchvision transformers rich
!git clone https://github.com/MohamedFarouk-1/SparseOpt.git
%cd SparseOpt
!pip install -e .

# Import the necessary modules
from sparseopt.huggingface import optimize_hf_model, load_hf_model, create_hf_input, benchmark_model
import torch

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load a model and tokenizer
model_name = "bert-base-uncased"
model, tokenizer = load_hf_model(model_name, device=device)

# Create example input
inputs = create_hf_input(tokenizer, "Hello, this is SparseOpt!", device=device)

# Benchmark the original model
original_results = benchmark_model(model, inputs, device=device, num_runs=10, warmup=3)
print(f"Original model latency: {original_results['mean_latency']:.2f}ms")

# Optimize the model
optimized_model, fusion_stats = optimize_hf_model(model_name, device=device)

# Benchmark the optimized model
optimized_results = benchmark_model(optimized_model, inputs, device=device, num_runs=10, warmup=3)
print(f"Optimized model latency: {optimized_results['mean_latency']:.2f}ms")

# Calculate speedup
speedup = original_results["mean_latency"] / optimized_results["mean_latency"]
print(f"Speedup: {speedup:.2f}x")
```

## Supported Models

SparseOpt has been tested with the following models:

- `bert-base-uncased`
- `distilbert-base-uncased`
- `facebook/opt-350m`

## Fusion Patterns

SparseOpt currently supports the following fusion patterns:

- **Linear + GELU**: Fuses linear layers followed by GELU activation
- **Linear + LayerNorm**: Fuses linear layers followed by layer normalization
- **MultiHeadAttention + LayerNorm**: Fuses multi-head attention followed by layer normalization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [PyTorch](https://pytorch.org/) for the FX framework
- [HuggingFace](https://huggingface.co/) for the transformer models 