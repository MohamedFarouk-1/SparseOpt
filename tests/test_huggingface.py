import unittest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparseopt.huggingface import (
    load_huggingface_model,
    create_model_input,
    trace_model,
    benchmark_model,
    print_benchmark_results
)
from sparseopt.graph.optimizer import GraphOptimizer
from sparseopt.graph.passes import LinearGELUFusion, ConvBatchNormReLUFusion

class TestHuggingFaceOptimization(unittest.TestCase):
    def setUp(self):
        # Use a small model for testing
        self.model_name = "gpt2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer
        self.model, self.tokenizer, self.model_info = load_huggingface_model(self.model_name)
        self.model = self.model.to(self.device)
        
        # Create input tensors
        self.inputs = create_model_input(self.tokenizer)
        self.inputs = {k: v.to(self.device) for k, v in self.inputs.items()}
        
        # Create optimizer
        self.optimizer = GraphOptimizer([
            LinearGELUFusion(),
            ConvBatchNormReLUFusion()
        ])
    
    def test_model_loading(self):
        """Test that model and tokenizer are loaded correctly."""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.tokenizer)
        self.assertIn("num_parameters", self.model_info)
        self.assertIn("num_layers", self.model_info)
    
    def test_model_tracing(self):
        """Test that model can be traced successfully."""
        traced_model, tracing_info = trace_model(self.model, self.inputs)
        self.assertTrue(tracing_info["success"])
        self.assertIsNotNone(traced_model)
    
    def test_optimization(self):
        """Test that optimization can be applied successfully."""
        # Trace model
        traced_model, _ = trace_model(self.model, self.inputs)
        
        # Apply optimization
        optimized_model, fusion_stats = self.optimizer.optimize(traced_model, self.inputs)
        
        # Verify fusion stats
        self.assertIsNotNone(fusion_stats)
        self.assertIn("num_fusions", fusion_stats)
    
    def test_numerical_correctness(self):
        """Test that optimization preserves numerical correctness."""
        # Trace and optimize model
        traced_model, _ = trace_model(self.model, self.inputs)
        optimized_model, _ = self.optimizer.optimize(traced_model, self.inputs)
        
        # Compare outputs
        with torch.no_grad():
            original_output = self.model(**self.inputs)
            optimized_output = optimized_model(**self.inputs)
            
            # Check that outputs are close
            self.assertTrue(
                torch.allclose(
                    original_output.last_hidden_state,
                    optimized_output.last_hidden_state,
                    rtol=1e-3,
                    atol=1e-3
                )
            )
    
    def test_benchmarking(self):
        """Test that benchmarking works correctly."""
        # Trace and optimize model
        traced_model, _ = trace_model(self.model, self.inputs)
        optimized_model, fusion_stats = self.optimizer.optimize(traced_model, self.inputs)
        
        # Run benchmarks
        original_results = benchmark_model(
            self.model,
            self.inputs,
            device=self.device,
            num_runs=10,  # Use small number for testing
            warmup=5
        )
        
        optimized_results = benchmark_model(
            optimized_model,
            self.inputs,
            device=self.device,
            num_runs=10,
            warmup=5
        )
        
        # Verify benchmark results
        self.assertIn("mean_latency", original_results)
        self.assertIn("mean_latency", optimized_results)
        self.assertIn("std_latency", original_results)
        self.assertIn("std_latency", optimized_results)

if __name__ == "__main__":
    unittest.main() 