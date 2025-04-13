"""Test suite for core graph optimization passes."""

import unittest
import torch
import torch.nn as nn
import torch.fx as fx
from sparseopt.graph.passes.linear_fusion import LinearReLUFusion, LinearGELUFusion
from sparseopt.graph.passes.conv_fusion import ConvBatchNormReLUFusion
from sparseopt.graph.passes.dead_node import DeadNodeEliminationPass
from sparseopt.graph.passes.reordering import NodeReorderingPass
from sparseopt.graph.passes.dropout_elimination import DropoutElimination

class TestCorePasses(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def test_linear_relu_fusion(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.linear2 = nn.Linear(20, 30)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x
                
        model = SimpleModel().to(self.device)
        example_input = torch.randn(1, 10, device=self.device)
        
        # Trace model
        traced = fx.symbolic_trace(model)
        
        # Apply fusion
        fusion_pass = LinearReLUFusion()
        optimized, stats = fusion_pass.apply(traced, (example_input,))
        
        # Verify fusion stats
        self.assertEqual(stats["fused_patterns"], 1)
        
        # Verify numerical correctness
        with torch.no_grad():
            original_output = model(example_input)
            optimized_output = optimized(example_input)
            self.assertTrue(torch.allclose(original_output, optimized_output))
            
    def test_linear_gelu_fusion(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.linear2 = nn.Linear(20, 30)
                self.gelu = nn.GELU()
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.gelu(x)
                x = self.linear2(x)
                return x
                
        model = SimpleModel().to(self.device)
        example_input = torch.randn(1, 10, device=self.device)
        
        # Trace model
        traced = fx.symbolic_trace(model)
        
        # Apply fusion
        fusion_pass = LinearGELUFusion()
        optimized, stats = fusion_pass.apply(traced, (example_input,))
        
        # Verify fusion stats
        self.assertEqual(stats["fused_patterns"], 1)
        
        # Verify numerical correctness
        with torch.no_grad():
            original_output = model(example_input)
            optimized_output = optimized(example_input)
            self.assertTrue(torch.allclose(original_output, optimized_output))
            
    def test_conv_bn_relu_fusion(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3)
                self.bn1 = nn.BatchNorm2d(16)
                self.relu = nn.ReLU()
                self.conv2 = nn.Conv2d(16, 32, 3)
                
            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.conv2(x)
                return x
                
        model = SimpleModel().to(self.device)
        example_input = torch.randn(1, 3, 32, 32, device=self.device)
        
        # Trace model
        traced = fx.symbolic_trace(model)
        
        # Apply fusion
        fusion_pass = ConvBatchNormReLUFusion()
        optimized, stats = fusion_pass.apply(traced, (example_input,))
        
        # Verify fusion stats
        self.assertEqual(stats["fused_patterns"], 1)
        
        # Verify numerical correctness
        with torch.no_grad():
            original_output = model(example_input)
            optimized_output = optimized(example_input)
            self.assertTrue(torch.allclose(original_output, optimized_output))
            
    def test_dead_node_elimination(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.linear2 = nn.Linear(20, 30)
                
            def forward(self, x):
                # Create dead node
                dead = self.linear1(x)
                # Only use linear2
                return self.linear2(x)
                
        model = SimpleModel().to(self.device)
        example_input = torch.randn(1, 10, device=self.device)
        
        # Trace model
        traced = fx.symbolic_trace(model)
        
        # Apply elimination
        elimination_pass = DeadNodeEliminationPass()
        optimized, stats = elimination_pass.apply(traced, (example_input,))
        
        # Verify elimination stats
        self.assertEqual(stats["eliminated_nodes"], 1)
        
        # Verify numerical correctness
        with torch.no_grad():
            original_output = model(example_input)
            optimized_output = optimized(example_input)
            self.assertTrue(torch.allclose(original_output, optimized_output))
            
    def test_node_reordering(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.linear2 = nn.Linear(20, 30)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x
                
        model = SimpleModel().to(self.device)
        example_input = torch.randn(1, 10, device=self.device)
        
        # Trace model
        traced = fx.symbolic_trace(model)
        
        # Apply reordering
        reordering_pass = NodeReorderingPass()
        optimized, stats = reordering_pass.apply(traced, (example_input,))
        
        # Verify reordering stats
        self.assertGreater(stats["reordered_nodes"], 0)
        
        # Verify numerical correctness
        with torch.no_grad():
            original_output = model(example_input)
            optimized_output = optimized(example_input)
            self.assertTrue(torch.allclose(original_output, optimized_output))
            
    def test_dropout_elimination(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.dropout = nn.Dropout(0.5)
                self.linear2 = nn.Linear(20, 30)
                
            def forward(self, x):
                x = self.linear1(x)
                x = self.dropout(x)
                x = self.linear2(x)
                return x
                
        model = SimpleModel().to(self.device)
        example_input = torch.randn(1, 10, device=self.device)
        
        # Trace model
        traced = fx.symbolic_trace(model)
        
        # Apply elimination
        elimination_pass = DropoutElimination()
        optimized, stats = elimination_pass.apply(traced, (example_input,))
        
        # Verify elimination stats
        self.assertEqual(stats["eliminated_dropouts"], 1)
        
        # Verify numerical correctness
        with torch.no_grad():
            original_output = model(example_input)
            optimized_output = optimized(example_input)
            self.assertTrue(torch.allclose(original_output, optimized_output))
            
    def test_multi_pass_optimization(self):
        class ComplexModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.linear2 = nn.Linear(20, 30)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.5)
                self.gelu = nn.GELU()
                
            def forward(self, x):
                # Create some dead nodes
                dead = self.linear1(x)
                x = self.linear2(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.gelu(x)
                return x
                
        model = ComplexModel().to(self.device)
        example_input = torch.randn(1, 10, device=self.device)
        
        # Trace model
        traced = fx.symbolic_trace(model)
        
        # Apply multiple passes
        passes = [
            LinearReLUFusion(),
            LinearGELUFusion(),
            DeadNodeEliminationPass(),
            DropoutElimination(),
            NodeReorderingPass()
        ]
        
        optimized = traced
        for pass_ in passes:
            optimized, _ = pass_.apply(optimized, (example_input,))
            
        # Verify numerical correctness
        with torch.no_grad():
            original_output = model(example_input)
            optimized_output = optimized(example_input)
            self.assertTrue(torch.allclose(original_output, optimized_output))

if __name__ == "__main__":
    unittest.main() 