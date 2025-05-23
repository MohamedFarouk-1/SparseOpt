"""
HuggingFace Model Support.

This module provides utilities for loading and optimizing HuggingFace models
using the SparseOpt toolkit.
"""

import torch
import torch.fx as fx
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    PreTrainedModel, PreTrainedTokenizer, GPT2PreTrainedModel, GPT2Model, GPT2LMHeadModel, GPTNeoForCausalLM
)
from typing import Dict, Any, Tuple, Optional, Union, List
from rich.console import Console
from rich.table import Table

console = Console()

class AttentionWrapper(torch.nn.Module):
    def __init__(self, attn_layer):
        super().__init__()
        self.attn = attn_layer
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.attn(hidden_states, attention_mask=attention_mask)
        if isinstance(outputs, tuple):
            return outputs[0]  # Return just the hidden states
        return outputs

class StaticModelWrapper(torch.nn.Module):
    """Wrapper for HuggingFace models to make them traceable with static shapes."""
    
    def __init__(self, model: PreTrainedModel, max_sequence_length: int = 512):
        super().__init__()
        self.model = model
        self.config = model.config
        self.max_sequence_length = max_sequence_length
        self.tokenizer = None
        
        # Create static inputs for tracing
        self.static_input_ids = torch.zeros((1, max_sequence_length), dtype=torch.long)
        self.static_attention_mask = torch.ones((1, max_sequence_length), dtype=torch.long)
        if torch.cuda.is_available():
            self.static_input_ids = self.static_input_ids.cuda()
            self.static_attention_mask = self.static_attention_mask.cuda()
            self.model = self.model.cuda()
        
    def set_tokenizer(self, tokenizer: PreTrainedTokenizer):
        """Set the tokenizer for the model."""
        self.tokenizer = tokenizer
        
    def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass that handles both string and tensor inputs."""
        if input_ids is None:
            input_ids = self.static_input_ids
        if attention_mask is None:
            attention_mask = self.static_attention_mask
            
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits if hasattr(outputs, "logits") else outputs.last_hidden_state

class HFTracer(fx.Tracer):
    """Custom tracer for HuggingFace models that handles leaf modules."""
    
    def __init__(self, leaf_modules: Optional[List[str]] = None):
        super().__init__()
        self.leaf_modules = leaf_modules or []
        
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        # Check if module class name is in leaf_modules
        if any(leaf in m.__class__.__name__ for leaf in self.leaf_modules):
            return True
        return super().is_leaf_module(m, module_qualified_name)

class TransformerBlockWrapper(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        # Wrap the attention layer if it exists
        if hasattr(block, 'attn'):
            block.attn = AttentionWrapper(block.attn)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.block(hidden_states, attention_mask=attention_mask)
        if isinstance(outputs, tuple):
            return outputs[0]  # Return just the hidden states
        return outputs

def load_hf_model(
    model_name: str, 
    model_type: str = "auto",
    device: str = "cuda",
    task: Optional[str] = None
) -> Tuple[torch.nn.Module, Any]:
    """
    Load a HuggingFace model and tokenizer.
    
    Args:
        model_name: Name of the model to load (e.g., "gpt2", "distilbert-base-uncased")
        model_type: Type of model to load ("auto", "causal", "sequence", etc.)
        device: Device to load the model on
        task: Optional task-specific model (e.g., "text-generation", "sequence-classification")
        
    Returns:
        Tuple of (model, tokenizer)
    """
    console.print(f"[bold blue]Loading {model_name} from HuggingFace...[/bold blue]")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model based on type
    if model_type == "auto":
        if task == "text-generation":
            model = AutoModelForCausalLM.from_pretrained(model_name)
        elif task == "sequence-classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        else:
            model = AutoModel.from_pretrained(model_name)
    elif model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif model_type == "sequence":
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
    
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    console.print(f"[bold green]Model loaded successfully![/bold green]")
    return model, tokenizer

def create_hf_input(
    tokenizer: Any,
    text: Union[str, List[str]] = "Hello, this is SparseOpt!",
    device: str = "cuda",
    max_length: int = 128,
    batch_size: int = 1
) -> Dict[str, torch.Tensor]:
    """
    Create input tensors for a HuggingFace model.
    
    Args:
        tokenizer: HuggingFace tokenizer
        text: Input text or list of texts
        device: Device to place tensors on
        max_length: Maximum sequence length
        batch_size: Batch size for inputs
        
    Returns:
        Dictionary of input tensors
    """
    # Handle single string or list of strings
    if isinstance(text, str):
        texts = [text] * batch_size
    else:
        texts = text
    
    # Tokenize inputs
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    return inputs

def trace_hf_model(model: torch.nn.Module, example_input: Union[str, Dict[str, torch.Tensor]], tracer: fx.Tracer) -> fx.GraphModule:
    """
    Trace a HuggingFace model using torch.fx.
    
    Args:
        model: The model to trace
        example_input: Example input for tracing (string or tokenized input dict)
        tracer: Custom tracer for HuggingFace models
        
    Returns:
        Traced model as a GraphModule
    """
    print("Tracing model...")
    
    # Trace the model
    try:
        graph = tracer.trace(model)
        graph_module = fx.GraphModule(model, graph)
        return graph_module
        
    except Exception as e:
        raise RuntimeError(f"Failed to trace model: {str(e)}")

def optimize_hf_model(
    model: Union[torch.nn.Module, str],
    example_inputs: Optional[Union[Dict[str, Any], Tuple[Any, ...], str]] = None,
    leaf_modules: Optional[List[str]] = None,
    device: Optional[str] = None,
    **kwargs
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Optimize a HuggingFace model using graph optimization passes.
    
    Args:
        model: The HuggingFace model to optimize (either a model instance or model name)
        example_inputs: Example inputs for tracing (optional - will be created if not provided)
        leaf_modules: List of module class names to treat as leaf modules during tracing.
                     Default includes common HuggingFace model classes.
        device: Device to run the model on (optional - will be auto-detected if not provided)
        **kwargs: Additional arguments passed to optimize_model
        
    Returns:
        Tuple of (optimized_model, optimization_statistics)
    """
    # Auto-detect device if not provided
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Handle model name string
    tokenizer = None
    if isinstance(model, str):
        model_name = model
        console.print(f"[bold blue]Loading {model_name} from HuggingFace...[/bold blue]")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        console.print(f"[bold green]Model loaded successfully![/bold green]")
    
    # Create example inputs if not provided
    if example_inputs is None and tokenizer is not None:
        example_inputs = create_model_input(tokenizer, "Hello, this is SparseOpt!")
        example_inputs = {k: v.to(device) for k, v in example_inputs.items()}
    elif isinstance(example_inputs, str) and tokenizer is not None:
        example_inputs = create_model_input(tokenizer, example_inputs)
        example_inputs = {k: v.to(device) for k, v in example_inputs.items()}
    
    # Default leaf modules for common HuggingFace models
    default_leaf_modules = [
        "DistilBertModel",
        "BertModel", 
        "GPT2Block",
        "GPT2Model",
        "GPTNeoBlock",
        "GPTNeoModel",
        "RobertaModel",
        "T5Model",
        "BartModel",
        "OPTModel",
        "OPTDecoder",
        "OPTDecoderLayer"
    ]
    
    # Use provided leaf modules or defaults
    leaf_modules = leaf_modules or default_leaf_modules
    
    # Create static model wrapper
    static_model = StaticModelWrapper(model)
    if tokenizer is not None:
        static_model.set_tokenizer(tokenizer)
    
    # Create custom tracer with leaf modules
    tracer = HFTracer(leaf_modules=leaf_modules)
    
    # Trace the model
    try:
        traced_model = trace_hf_model(static_model, example_inputs, tracer)
    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] FX tracing failed with error: {str(e)}")
        console.print("[yellow]Falling back to static model without tracing[/yellow]")
        traced_model = static_model
    
    # Create optimizer
    optimizer = ModelOptimizer()
    
    # Apply fusion passes
    fusion_patterns = [
        LinearGELUFusion(),  # Linear + GELU fusion
        LinearLayerNormFusion(),  # Linear + LayerNorm fusion
        MHALayerNormFusion()  # MultiHeadAttention + LayerNorm fusion
    ]
    
    # Run optimization passes
    optimized_model = optimizer.optimize(
        traced_model,
        fusion_patterns=fusion_patterns,
        layer_by_layer=True  # Apply optimizations layer by layer for better handling of transformer blocks
    )
    
    # Get optimization statistics
    stats = {
        "num_fusions": sum(len(result.fusions) for result in optimizer.fusion_results.values()),
        "fusion_results": optimizer.fusion_results
    }
    
    return optimized_model, stats

def create_model_input(
    tokenizer: Any,
    text: str = "Hello, world!",
    max_length: int = 128
) -> Dict[str, torch.Tensor]:
    """
    Create input tensors for the model.
    
    Args:
        tokenizer: HuggingFace tokenizer
        text: Input text to tokenize
        max_length: Maximum sequence length
        
    Returns:
        Dictionary of input tensors
    """
    # Tokenize input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True
    )
    
    return inputs

def benchmark_model(
    model: Any,
    inputs: Dict[str, torch.Tensor],
    device: str = "cuda",
    num_runs: int = 100,
    warmup: int = 10
) -> Dict[str, float]:
    """
    Benchmark model performance.
    
    Args:
        model: Model to benchmark
        inputs: Dictionary of input tensors
        device: Device to run on
        num_runs: Number of benchmark runs
        warmup: Number of warmup runs
        
    Returns:
        Dictionary of benchmark results
    """
    # Warmup runs
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(**inputs)
    
    # Benchmark runs
    latencies = []
    for _ in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        with torch.no_grad():
            _ = model(**inputs)
        end.record()
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        latencies.append(start.elapsed_time(end))
    
    # Calculate statistics
    latencies = torch.tensor(latencies)
    mean_latency = latencies.mean().item()
    std_latency = latencies.std().item()
    
    return {
        "mean_latency": mean_latency,
        "std_latency": std_latency,
        "min_latency": latencies.min().item(),
        "max_latency": latencies.max().item()
    }

def print_benchmark_results(
    original_results: Dict[str, float],
    optimized_results: Dict[str, float],
    fusion_stats: Dict[str, Any]
) -> None:
    """
    Print benchmark results in a formatted table.
    
    Args:
        original_results: Results from original model
        optimized_results: Results from optimized model
        fusion_stats: Statistics about fusion operations
    """
    # Create results table
    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Original", style="green")
    table.add_column("Optimized", style="green")
    table.add_column("Speedup", style="yellow")
    
    # Add latency results
    table.add_row(
        "Mean Latency (ms)",
        f"{original_results['mean_latency']:.2f}",
        f"{optimized_results['mean_latency']:.2f}",
        f"{original_results['mean_latency'] / optimized_results['mean_latency']:.2f}x"
    )
    
    table.add_row(
        "Std Latency (ms)",
        f"{original_results['std_latency']:.2f}",
        f"{optimized_results['std_latency']:.2f}",
        "-"
    )
    
    # Add fusion stats
    table.add_row(
        "Number of Fusions",
        "-",
        str(fusion_stats.get("num_fusions", 0)),
        "-"
    )
    
    console.print(table) 