"""End-to-end tests for SparseOpt optimization and fusion logic."""

import os
import sys
import time
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from rich.console import Console
from rich.table import Table
import numpy as np
import unittest
from sparseopt import ModelOptimizer
from sparseopt.benchmark import ModelBenchmarker

# Add the parent directory to the path so we can import sparseopt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

console = Console()

class TestOptimization(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.warmup_runs = 5
        self.timed_runs = 20
        
    def test_resnet18(self):
        print("\nTesting ResNet18 Optimization")
        # Load model
        model = models.resnet18(pretrained=True)
        model = model.to(self.device)
        
        # Create dummy input
        input_data = torch.randn(1, 3, 224, 224, device=self.device)
        
        # Benchmark before optimization
        print("\nBenchmarking before optimization:")
        benchmarker = ModelBenchmarker(model, input_data, self.device)
        latency_before = benchmarker.measure_latency()
        
        # Optimize model
        optimizer = ModelOptimizer(model, device=self.device)
        optimized_model = optimizer.optimize()
        
        # Benchmark after optimization
        print("\nBenchmarking after optimization:")
        benchmarker = ModelBenchmarker(optimized_model, input_data, self.device)
        latency_after = benchmarker.measure_latency()
        
        # Compare outputs
        with torch.no_grad():
            output_before = model(input_data)
            output_after = optimized_model(input_data)
        
        # Check outputs match
        self.assertTrue(torch.allclose(output_before, output_after, rtol=1e-3, atol=1e-3))
        
        # Calculate speedup
        speedup = latency_before / latency_after
        print(f"\nResNet18 Speedup: {speedup:.2f}x")
        
        return speedup, True
        
    def test_gpt2(self):
        print("\nTesting GPT2 Optimization")
        # Load model and tokenizer
        model_name = "gpt2"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = model.to(self.device)
        
        # Create input
        input_text = "Hello, how are you?"
        inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        # Benchmark before optimization
        print("\nBenchmarking before optimization:")
        benchmarker = ModelBenchmarker(model, input_ids, self.device)
        latency_before = benchmarker.measure_latency()
        
        # Optimize model
        optimizer = ModelOptimizer(model, device=self.device)
        optimized_model = optimizer.optimize()
        
        # Benchmark after optimization
        print("\nBenchmarking after optimization:")
        benchmarker = ModelBenchmarker(optimized_model, input_ids, self.device)
        latency_after = benchmarker.measure_latency()
        
        # Compare outputs
        with torch.no_grad():
            output_before = model(input_ids)
            output_after = optimized_model(input_ids)
        
        # Check outputs match
        self.assertTrue(torch.allclose(output_before.logits, output_after.logits, rtol=1e-3, atol=1e-3))
        
        # Calculate speedup
        speedup = latency_before / latency_after
        print(f"\nGPT2 Speedup: {speedup:.2f}x")
        
        return speedup, True
        
    def test_bert(self):
        print("\nTesting BERT Optimization")
        # Load model and tokenizer
        model_name = "bert-base-uncased"
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = model.to(self.device)
        
        # Create input
        input_text = "Hello, how are you?"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Benchmark before optimization
        print("\nBenchmarking before optimization:")
        benchmarker = ModelBenchmarker(model, (input_ids, attention_mask), self.device)
        latency_before = benchmarker.measure_latency()
        
        # Optimize model
        optimizer = ModelOptimizer(model, device=self.device)
        optimized_model = optimizer.optimize()
        
        # Benchmark after optimization
        print("\nBenchmarking after optimization:")
        benchmarker = ModelBenchmarker(optimized_model, (input_ids, attention_mask), self.device)
        latency_after = benchmarker.measure_latency()
        
        # Compare outputs
        with torch.no_grad():
            output_before = model(input_ids, attention_mask=attention_mask)
            output_after = optimized_model(input_ids, attention_mask=attention_mask)
        
        # Check outputs match
        self.assertTrue(torch.allclose(output_before.last_hidden_state, output_after.last_hidden_state, rtol=1e-3, atol=1e-3))
        
        # Calculate speedup
        speedup = latency_before / latency_after
        print(f"\nBERT Speedup: {speedup:.2f}x")
        
        return speedup, True

if __name__ == '__main__':
    print("\nRunning SparseOpt End-to-End Tests\n")
    
    test = TestOptimization()
    test.setUp()
    
    results = []
    
    # Run tests
    try:
        results.append(("ResNet18", *test.test_resnet18()))
    except Exception as e:
        print(f"Error testing ResNet18: {str(e)}")
        results.append(("ResNet18", 0.0, False))
        
    try:
        results.append(("GPT2", *test.test_gpt2()))
    except Exception as e:
        print(f"Error testing GPT2: {str(e)}")
        results.append(("GPT2", 0.0, False))
        
    try:
        results.append(("BERT", *test.test_bert()))
    except Exception as e:
        print(f"Error testing BERT: {str(e)}")
        results.append(("BERT", 0.0, False))
    
    # Print summary table
    print("\nTest Summary")
    print("     Optimization Test Results     ")
    print("┏━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━┓")
    print("┃ Model ┃ Speedup ┃ Outputs Match ┃")
    print("┡━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━┩")
    for model, speedup, outputs_match in results:
        print(f"┃ {model:<7} ┃ {speedup:>7.2f}x ┃ {str(outputs_match):<13} ┃")
    print("└───────┴─────────┴───────────────┘") 