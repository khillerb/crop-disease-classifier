import os

# Fix OpenMP duplicate library issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
import sys
import json

sys.path.append(".")
from model import AlexNetCBAM


def quantize_model(checkpoint_path, num_classes, output_name):
    """Quantize a single model"""
    print(f"\nProcessing {output_name}...")

    # Load model
    model = AlexNetCBAM(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create example input
    example = torch.rand(1, 3, 224, 224)

    # Trace the model
    traced_script_module = torch.jit.trace(model, example)

    # Quantize for mobile
    quantized_model = torch.quantization.quantize_dynamic(
        traced_script_module, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Optimize for mobile
    optimized_model = optimize_for_mobile(quantized_model)

    # Save
    os.makedirs("android_models", exist_ok=True)
    output_path = f"android_models/{output_name}.ptl"
    optimized_model._save_for_lite_interpreter(output_path)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"✓ {output_name}.ptl saved ({size_mb:.2f} MB)")

    return output_path


def main():
    # Create output directory
    os.makedirs("android_models", exist_ok=True)

    # Disease label mappings
    labels = {
        "soy": {"bacterial blight": 0, "frogeye": 1, "healthy": 2, "soy rust": 3},
        "corn": {"leaf blight": 0, "common rust": 1, "gray leaf spot": 2, "healthy": 3},
        "rice": {
            "bacterial blight": 0,
            "blast": 1,
            "brown spot": 2,
            "healthy": 3,
            "tungro": 4,
        },
        "potato": {
            "Bacteria": 0,
            "Fungi": 1,
            "Healthy": 2,
            "Nematode": 3,
            "Pest": 4,
            "Phytopthora": 5,
            "Virus": 6,
        },
        "wheat": {
            "BlackPoint": 0,
            "FusariumFootRot": 1,
            "HealthyLeaf": 2,
            "LeafBlight": 3,
            "WheatBlast": 4,
        },
        "all": {"soy": 0, "corn": 1, "rice": 2, "potato": 3, "wheat": 4},
    }

    # Save labels for Android app
    with open("android_models/labels.json", "w") as f:
        json.dump(labels, f, indent=2)
    print("✓ labels.json created")

    # Model configurations: (checkpoint_path, num_classes, output_name)
    models = [
        ("outputs/all_plants/checkpoints/best_model.pth", 5, "plant_classifier"),
        ("outputs/soy/checkpoints/best_model.pth", 4, "soy_disease"),
        ("outputs/corn/checkpoints/best_model.pth", 4, "corn_disease"),
        ("outputs/rice/checkpoints/best_model.pth", 5, "rice_disease"),
        ("outputs/potato/checkpoints/best_model.pth", 7, "potato_disease"),
        ("outputs/wheat/checkpoints/best_model.pth", 5, "wheat_disease"),
    ]

    print("Starting model quantization...")
    print("=" * 60)

    for checkpoint_path, num_classes, output_name in models:
        if os.path.exists(checkpoint_path):
            quantize_model(checkpoint_path, num_classes, output_name)
        else:
            print(f"⚠ Warning: {checkpoint_path} not found, skipping...")

    print("\n" + "=" * 60)
    print("✓ All models quantized successfully!")
    print("\nNext steps:")
    print("1. Copy all .ptl files from android_models/ to app/src/main/assets/models/")
    print("2. Copy labels.json to app/src/main/assets/")
    print("3. Build and run the Android app")


if __name__ == "__main__":
    main()
