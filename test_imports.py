import sys
print('Python:', sys.version)

tests = [
    ('torch + CUDA', 'torch + CUDA', lambda: __import__('torch') and print(__import__('torch').__version__, __import__('torch').cuda.get_device_name(0) if __import__('torch').cuda.is_available() else 'CPU')),
]

import torch
print(f"  {'OK':4}  torch+CUDA: {torch.__version__}, CUDA={torch.cuda.is_available()}, GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

import monai
print(f"  OK    monai: {monai.__version__}")

import nibabel
print(f"  OK    nibabel: {nibabel.__version__}")

import SimpleITK
print(f"  OK    SimpleITK: {SimpleITK.__version__}")

import pydicom
print(f"  OK    pydicom: {pydicom.__version__}")

import einops
print(f"  OK    einops: {einops.__version__}")

import matplotlib
print(f"  OK    matplotlib: {matplotlib.__version__}")

import numpy
print(f"  OK    numpy: {numpy.__version__}")

try:
    import dicom2nifti
    print(f"  OK    dicom2nifti: installed")
except Exception as e:
    print(f"  FAIL  dicom2nifti: {e}")

# Test MONAI transforms
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Orientationd, Spacingd, ScaleIntensityRanged, CropForegroundd, ToTensord
)
print("  OK    MONAI transforms: all imported")

from monai.inferers import sliding_window_inference
print("  OK    MONAI sliding_window_inference: imported")

print("\nAll critical imports successful!")
