"""
Google Colab script for testing SparseOpt HuggingFace optimization on GPU.

This script demonstrates how to:
1. Install necessary libraries
2. Clone the SparseOpt repository
3. Load a HuggingFace model
4. Apply SparseOpt optimization
5. Benchmark performance
6. Verify numerical correctness
"""

# 1. Install necessary libraries
!pip install torch torchvision transformers rich

# 2. Clone SparseOpt repo (replace with your actual repo if on GitHub)
!git clone https://github.com/YOUR_USERNAME/SparseOpt.git
!cd SparseOpt

# 3. Set device to GPU (assert CUDA is available)
import torch
assert torch.cuda.is_available(), "CUDA is not available"
device = torch.device("cuda")
print(f"Using device: {device}")

# 4. Load a HuggingFace model + tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
print(f"Loading {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 5. Create example input
example_input = "Hello world! This is a test of SparseOpt optimization."
print(f"Example input: {example_input}")

# 6. Run baseline inference
model.eval()
with torch.no_grad():
    torch.cuda.synchronize()
    import time; start = time.time()
    inputs = tokenizer(example_input, return_tensors="pt").to(device)
    _ = model(**inputs)
    torch.cuda.synchronize()
    baseline = time.time() - start

print(f"Baseline GPU inference time: {baseline:.4f} seconds")

# 7. Import and run SparseOpt optimizer
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

# 8. Run optimized inference
optimized_model.eval()
with torch.no_grad():
    torch.cuda.synchronize()
    start = time.time()
    _ = optimized_model(**inputs)
    torch.cuda.synchronize()
    optimized = time.time() - start

print(f"Optimized GPU inference time: {optimized:.4f} seconds")
print(f"Speedup: {baseline / optimized:.2f}x")

# 9. Print optimization statistics
print("\nOptimization Statistics:")
for pass_name, pass_stats in stats.items():
    print(f"\n{pass_name}:")
    if isinstance(pass_stats, dict):
        for stat_name, value in pass_stats.items():
            print(f"  {stat_name}: {value}")
    else:
        print(f"  {pass_stats}")

# 10. Validate numerical correctness
print("\nValidating numerical correctness...")
with torch.no_grad():
    original_output = model(**inputs).logits
    optimized_output = optimized_model(**inputs).logits

is_close = torch.allclose(original_output, optimized_output, atol=1e-5)
print(f"Numerical Correctness: {is_close}")

# 11. Run more detailed benchmarking
print("\nRunning detailed benchmarking...")
from sparseopt.huggingface import benchmark_model

# Benchmark original model
original_results = benchmark_model(
    model=model,
    inputs=inputs,
    device="cuda",
    num_runs=50,
    warmup=5
)

# Benchmark optimized model
optimized_results = benchmark_model(
    model=optimized_model,
    inputs=inputs,
    device="cuda",
    num_runs=50,
    warmup=5
)

# Print benchmark results
print("\nBenchmark Results:")
print(f"Original - Mean: {original_results['mean_latency']:.2f}ms, Std: {original_results['std_latency']:.2f}ms")
print(f"Optimized - Mean: {optimized_results['mean_latency']:.2f}ms, Std: {optimized_results['std_latency']:.2f}ms")
print(f"Speedup: {original_results['mean_latency'] / optimized_results['mean_latency']:.2f}x")

print("\nOptimization complete!") 