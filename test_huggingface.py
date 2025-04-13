#!/usr/bin/env python
"""
Simple script to run the HuggingFace end-to-end test directly.
This script can be run from the command line to test the HuggingFace optimization pipeline.
"""

import os
import sys
import argparse

# Add parent directory to path to import sparseopt
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the end-to-end test function
from tests.test_huggingface_e2e import run_end_to_end_test

def main():
    """Main entry point for the HuggingFace end-to-end test."""
    parser = argparse.ArgumentParser(description="End-to-end test for HuggingFace model optimization")
    parser.add_argument("--model", type=str, default="gpt2", help="HuggingFace model name")
    parser.add_argument("--text", type=str, default="Hello, this is SparseOpt!", help="Sample text for input")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda/cpu)")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for numerical correctness")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for numerical correctness")
    
    args = parser.parse_args()
    
    success = run_end_to_end_test(
        model_name=args.model,
        sample_text=args.text,
        device=args.device,
        num_runs=args.num_runs,
        warmup=args.warmup,
        rtol=args.rtol,
        atol=args.atol
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 