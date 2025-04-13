import torch
import torchvision.models as models
from fusion_pass import optimize_model, print_fusion_results

def main():
    print("🚀 Running optimization on ResNet18")

    # Load model
    model = models.resnet18(weights=None)

    # Measure latency before optimization
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        model.eval()
        before = []
        for _ in range(10):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(dummy_input)
            end.record()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            before.append(start.elapsed_time(end))
        print(f"⏱️ Latency before optimization: {sum(before)/len(before):.2f} ms")

    # Optimize model
    optimized_model, fusion_results = optimize_model(model)

    # Print fusion results
    print_fusion_results(fusion_results)

    # Measure latency after optimization
    with torch.no_grad():
        optimized_model.eval()
        after = []
        for _ in range(10):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            optimized_model(dummy_input)
            end.record()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            after.append(start.elapsed_time(end))
        print(f"✅ Latency after optimization: {sum(after)/len(after):.2f} ms")

if __name__ == "__main__":
    main()
