# HuggingFace Model Optimization Test

This document provides instructions on how to run the end-to-end test for HuggingFace model optimization in SparseOpt.

## Overview

The end-to-end test validates the entire optimization pipeline for HuggingFace models:

1. ✅ Model loading and tracing
2. ✅ Pattern fusion (Linear-GELU, Conv-BN-ReLU)
3. ✅ Numerical correctness verification
4. ✅ Performance benchmarking
5. ✅ Detailed reporting

## Requirements

- Python 3.7+
- PyTorch 1.8+
- Transformers library
- Rich library for formatting
- CUDA (optional, for GPU acceleration)

## Running the Test

### Option 1: Using the CLI

```bash
# Run with default settings (GPT-2 model)
python -m sparseopt.cli test-huggingface

# Run with custom settings
python -m sparseopt.cli test-huggingface --model distilbert-base-uncased --text "Hello, this is SparseOpt!" --device cuda
```

### Option 2: Using the Direct Script

```bash
# Run with default settings (GPT-2 model)
./test_huggingface.py

# Run with custom settings
./test_huggingface.py --model distilbert-base-uncased --text "Hello, this is SparseOpt!" --device cuda
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model`, `-m` | HuggingFace model name | `gpt2` |
| `--text`, `-t` | Sample text for input | `"Hello, this is SparseOpt!"` |
| `--device`, `-d` | Device to run on (cuda/cpu) | `None` (auto-detect) |
| `--num-runs`, `-n` | Number of benchmark runs | `100` |
| `--warmup`, `-w` | Number of warmup runs | `10` |
| `--rtol` | Relative tolerance for numerical correctness | `1e-3` |
| `--atol` | Absolute tolerance for numerical correctness | `1e-3` |
| `--debug` | Show full traceback on errors | `False` |

## Example Output

```
╭──────────────────────────────────────────────────────────────────────────────╮
│                     Running end-to-end test for gpt2 on CUDA                  │
╰──────────────────────────────────────────────────────────────────────────────╯

╭──────────────────────────────────────────────────────────────────────────────╮
│                              Model Information                               │
├──────────────┬──────────────────────────────────────────────────────────────┤
│ Property     │ Value                                                         │
├──────────────┼──────────────────────────────────────────────────────────────┤
│ Model        │ gpt2                                                          │
│ Parameters   │ 124,439,808                                                  │
│ Layers       │ 12                                                           │
│ Device       │ CUDA                                                         │
╰──────────────┴──────────────────────────────────────────────────────────────╯

✓ Model traced successfully with 156 nodes
✓ Applied 24 fusion patterns
✓ Numerical correctness verified

╭──────────────────────────────────────────────────────────────────────────────╮
│                              Benchmark Results                               │
├──────────────┬──────────┬──────────┬──────────┤
│ Metric       │ Original │ Optimized│ Speedup  │
├──────────────┼──────────┼──────────┼──────────┤
│ Mean Latency │ 12.45 ms │ 8.32 ms  │ 1.50x    │
│ Std Latency  │ 0.32 ms  │ 0.21 ms  │ -        │
│ Fusions      │ -        │ 24       │ -        │
╰──────────────┴──────────┴──────────┴──────────╯

╭──────────────────────────────────────────────────────────────────────────────╮
│ ✅ All checks passed!                                                         │
│                                                                              │
│ Model: gpt2                                                                  │
│ Device: CUDA                                                                 │
│ Fusion patterns: 24                                                          │
│ Speedup: 1.50x                                                               │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Troubleshooting

### Common Issues

1. **CUDA not available**: If you don't have CUDA installed or available, the test will run on CPU by default.

2. **Model loading errors**: Make sure you have the Transformers library installed and have internet access to download the model.

3. **Tracing errors**: Some models may have dynamic control flow that cannot be traced by torch.fx. In this case, the test will fail with a tracing error.

4. **Numerical correctness failures**: If the numerical correctness check fails, you may need to adjust the `--rtol` and `--atol` parameters to allow for more tolerance.

### Debug Mode

To get more detailed error information, run the test with the `--debug` flag:

```bash
python -m sparseopt.cli test-huggingface --debug
```

## Adding to CI/CD

To add this test to your CI/CD pipeline, you can use the following command:

```bash
python -m sparseopt.cli test-huggingface --model gpt2 --num-runs 10 --warmup 2
```

This will run a quick test with fewer benchmark runs to speed up the CI/CD process. 