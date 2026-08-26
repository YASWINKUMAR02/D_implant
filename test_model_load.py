import sys
import os
from pathlib import Path

# Add OralSeg to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "OralSeg"))

checkpoint_path = str(ROOT_DIR / "models" / "model_workstation39.pt")
print(f"Testing OralSeg model loading with checkpoint: {checkpoint_path}")

try:
    from utils.oralseg_inference import load_model
    if os.path.exists(checkpoint_path):
        print("Checkpoint file exists! Calling load_model()...")
        model, device, device_name = load_model(checkpoint_path)
        print(f"SUCCESS: Successfully loaded OralSeg model on {device_name} ({device})!")
        
        # Test a dummy forward pass
        import torch
        dummy_input = torch.randn(1, 1, 96, 96, 96, device=device)
        print(f"Running dummy forward pass with input shape {dummy_input.shape}...")
        with torch.no_grad():
            output = model(dummy_input)
        print(f"SUCCESS: Forward pass successful! Output logits shape: {output.shape}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
except Exception as e:
    import traceback
    print(f"Error during load_model: {e}")
    traceback.print_exc()
