# Simplified SparseOpt HuggingFace Optimization on Google Colab

This guide explains how to use the simplified SparseOpt script to optimize HuggingFace models on Google Colab.

## Setup Instructions

1. **Create a new Google Colab notebook**

2. **Copy the following code into a cell and run it:**

```python
# Install necessary libraries
!pip install torch torchvision transformers rich

# Clone SparseOpt repo (replace with your actual repo if on GitHub)
!git clone https://github.com/YOUR_USERNAME/SparseOpt.git
!cd SparseOpt

# Set device to GPU (assert CUDA is available)
import torch
assert torch.cuda.is_available(), "CUDA is not available"
device = torch.device("cuda")
print(f"Using device: {device}")

# Load a HuggingFace model + tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
print(f"Loading {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create example input
example_input = "Hello world! This is a test of SparseOpt optimization."
print(f"Example input: {example_input}")

# Run baseline inference
model.eval()
with torch.no_grad():
    torch.cuda.synchronize()
    import time; start = time.time()
    inputs = tokenizer(example_input, return_tensors="pt").to(device)
    _ = model(**inputs)
    torch.cuda.synchronize()
    baseline = time.time() - start

print(f"Baseline GPU inference time: {baseline:.4f} seconds")

# Import and run SparseOpt optimizer
from sparseopt.huggingface import optimize_hf_model

print("Optimizing model...")
optimized_model, stats = optimize_hf_model(
    model=model,
    tokenizer=tokenizer,
    example_input=example_input,
    max_sequence_length=128,
    device="cuda"
)

# Check if optimization failed
if optimized_model is None:
    print("\nOptimization failed. Check the error message above.")
    exit()

# Run optimized inference
optimized_model.eval()
with torch.no_grad():
    torch.cuda.synchronize()
    start = time.time()
    _ = optimized_model(**inputs)
    torch.cuda.synchronize()
    optimized = time.time() - start

print(f"Optimized GPU inference time: {optimized:.4f} seconds")
print(f"Speedup: {baseline / optimized:.2f}x")

# Print optimization statistics
print("\nOptimization Statistics:")
for pass_name, pass_stats in stats.items():
    print(f"\n{pass_name}:")
    if isinstance(pass_stats, dict):
        for stat_name, value in pass_stats.items():
            print(f"  {stat_name}: {value}")
    else:
        print(f"  {pass_stats}")

# Validate numerical correctness
print("\nValidating numerical correctness...")
with torch.no_grad():
    original_output = model(**inputs).logits
    optimized_output = optimized_model(**inputs).logits

is_close = torch.allclose(original_output, optimized_output, atol=1e-5)
print(f"Numerical Correctness: {is_close}")

print("\nOptimization complete!")
```

## What This Script Does

1. **Installation**: Installs PyTorch, Transformers, and other required libraries
2. **Repository Setup**: Clones the SparseOpt repository
3. **Model Loading**: Loads a GPT-2 model from HuggingFace
4. **Baseline Measurement**: Measures the inference time of the original model
5. **Optimization**: Applies SparseOpt's graph optimization passes to the model
6. **Performance Comparison**: Measures the inference time of the optimized model
7. **Correctness Verification**: Ensures the optimized model produces the same outputs

## Expected Results

You should see:
- A baseline inference time for the original model
- An optimized inference time for the SparseOpt-optimized model
- A speedup factor showing how much faster the optimized model is
- Statistics about the optimization passes applied
- Verification that the outputs are numerically correct

## Troubleshooting

If you encounter issues:

1. **CUDA Not Available**: Make sure you're using a GPU runtime in Colab
2. **Import Errors**: Check that all dependencies are installed correctly
3. **Optimization Failures**: The script will print error messages if optimization fails
4. **Numerical Correctness Issues**: If the outputs don't match, try adjusting the tolerance

## Notes

- The script uses GPT-2 as an example, but you can modify it to use other HuggingFace models
- The optimization process may take some time depending on the model size
- For larger models, you may need to adjust the `max_sequence_length` parameter 