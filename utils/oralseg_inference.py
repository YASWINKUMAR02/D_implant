"""
oralseg_inference.py
Loads the official OralSeg model (model_workstation39.pt) and runs 3D inference
using MONAI sliding_window_inference. All preprocessing follows the official
OralSeg pipeline (MONAI BTCV pattern).

Architecture: Hybrid SwinUNETR + MambaEncoder
  - in_channels=1, out_channels=36 (background + 35 anatomical classes)
  - feature_size=48, depths=(2,2,2,2)
  - feat_size=[48, 96, 192, 384]

Label mapping (from OralSeg dataset.json):
  0  = background
  1  = maxilla
  2  = mandible
  3  = tooth 11  ...  10 = tooth 18
  11 = tooth 21  ...  18 = tooth 28
  19 = tooth 31  ...  26 = tooth 38
  27 = tooth 41  ...  34 = tooth 48
  35 = mandibular canal
"""

import os
import sys
import gc
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple

import numpy as np
import torch
import nibabel as nib

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Ensure OralSeg source is on the Python path
# ─────────────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent.parent   # c:\D_Implant
_ORALSEG_DIR = _HERE / "OralSeg"

if str(_ORALSEG_DIR) not in sys.path:
    sys.path.insert(0, str(_ORALSEG_DIR))

# ─────────────────────────────────────────────────────────────────────────────
# OralSeg Label Map
# ─────────────────────────────────────────────────────────────────────────────
LABEL_MAP = {
    0:  "Background",
    1:  "Maxilla",
    2:  "Mandible",
    3:  "Tooth 11",
    4:  "Tooth 12",
    5:  "Tooth 13",
    6:  "Tooth 14",
    7:  "Tooth 15",
    8:  "Tooth 16",
    9:  "Tooth 17",
    10: "Tooth 18",
    11: "Tooth 21",
    12: "Tooth 22",
    13: "Tooth 23",
    14: "Tooth 24",
    15: "Tooth 25",
    16: "Tooth 26",
    17: "Tooth 27",
    18: "Tooth 28",
    19: "Tooth 31",
    20: "Tooth 32",
    21: "Tooth 33",
    22: "Tooth 34",
    23: "Tooth 35",
    24: "Tooth 36",
    25: "Tooth 37",
    26: "Tooth 38",
    27: "Tooth 41",
    28: "Tooth 42",
    29: "Tooth 43",
    30: "Tooth 44",
    31: "Tooth 45",
    32: "Tooth 46",
    33: "Tooth 47",
    34: "Tooth 48",
    35: "Mandibular Canal",
}

# FDI tooth label index → FDI tooth number
TOOTH_FDI_MAP = {
    3: 11,  4: 12,  5: 13,  6: 14,  7: 15,  8: 16,  9: 17,  10: 18,
    11: 21, 12: 22, 13: 23, 14: 24, 15: 25, 16: 26, 17: 27, 18: 28,
    19: 31, 20: 32, 21: 33, 22: 34, 23: 35, 24: 36, 25: 37, 26: 38,
    27: 41, 28: 42, 29: 43, 30: 44, 31: 45, 32: 46, 33: 47, 34: 48,
}


def _get_device() -> torch.device:
    """Select CUDA if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_model(device: torch.device):
    """
    Instantiate the OralSeg model with the exact configuration used during training.
    img_size must be divisible by 2^5=32. We use (96,96,96) for sliding window patches.
    """
    if str(_ORALSEG_DIR) not in sys.path:
        sys.path.insert(0, str(_ORALSEG_DIR))
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))

    try:
        from OralSeg.modified.monai.networks.nets.OralSeg import OralSeg
    except ImportError:
        from modified.monai.networks.nets.OralSeg import OralSeg

    model = OralSeg(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=36,
        feature_size=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        norm_name="instance",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        normalize=True,
        use_checkpoint=False,
        spatial_dims=3,
        downsample="merging",
        use_v2=False,
        in_chans=1,
        out_chans=36,
        feat_size=[48, 96, 192, 384],
        hidden_size=768,
        conv_block=True,
        res_block=True,
    )
    return model.to(device)


def load_model(checkpoint_path: str, progress_callback: Optional[Callable] = None) -> Tuple[Any, torch.device, str]:
    """
    Load OralSeg model from checkpoint.

    Returns:
        (model, device, device_name_str)

    Raises:
        FileNotFoundError if checkpoint is missing.
        RuntimeError on architecture mismatch.
    """
    def log(msg):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"OralSeg model checkpoint not found. Expected: {checkpoint_path}"
        )

    device = _get_device()
    device_name = "CPU"
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
        log(f"Using GPU: {device_name}")
    else:
        log("CUDA unavailable — running on CPU. Inference will be slow.")

    log("Building OralSeg architecture...")
    model = _build_model(device)

    log(f"Loading checkpoint: {os.path.basename(checkpoint_path)}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "net" in checkpoint:
            state_dict = checkpoint["net"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Strip DataParallel / DDP "module." prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith("module.") else k
        new_state_dict[new_key] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys in checkpoint: {len(missing)}")
    if unexpected:
        logger.warning(f"Unexpected keys in checkpoint: {len(unexpected)}")

    log("Model loaded successfully ✓")
    model.eval()
    return model, device, device_name


def _build_preprocessing_transforms():
    """
    Official OralSeg preprocessing transforms (MONAI BTCV standard):
    - Orientation: RAS
    - Spacing: 1.0mm isotropic
    - Intensity: ScaleIntensityRange (-1000 to 3000 HU → 0–1)
    - CropForeground
    """
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        Orientationd,
        Spacingd,
        ScaleIntensityRanged,
        CropForegroundd,
        ToTensord,
    )

    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode="bilinear",
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000,
            a_max=3000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(
            keys=["image"],
            source_key="image",
        ),
        ToTensord(keys=["image"]),
    ])


def run_inference(
    model,
    device: torch.device,
    nifti_path: str,
    output_seg_path: str,
    sw_batch_size: int = 2,
    overlap: float = 0.5,
    progress_callback: Optional[Callable] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run OralSeg sliding-window inference on a NIfTI volume.

    Args:
        model: loaded OralSeg model (already on device, eval mode)
        device: torch.device
        nifti_path: path to the preprocessed-compatible .nii.gz volume
        output_seg_path: where to save the segmentation NIfTI
        sw_batch_size: sliding window batch size (2 for 4GB VRAM)
        overlap: patch overlap fraction (0.5 recommended)
        progress_callback: optional progress function

    Returns:
        (seg_array_xyz, result_info)
    """
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd,
        Orientationd, Spacingd, ScaleIntensityRanged,
        CropForegroundd, ToTensord,
    )

    def log(msg):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    # ── 1. Preprocess ────────────────────────────────────────────────────────
    log("Applying OralSeg preprocessing transforms...")
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=3000,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > 0.02, margin=4),
        ToTensord(keys=["image"]),
    ])

    data = transforms({"image": nifti_path})
    image_tensor = data["image"]  # shape: (1, D, H, W) after channel-first

    log(f"Preprocessed volume shape: {list(image_tensor.shape)}")
    log(f"Intensity range after normalization: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")

    # Add batch dimension: (1, 1, D, H, W)
    input_tensor = image_tensor.unsqueeze(0).to(device)

    # ── 2. Inference ─────────────────────────────────────────────────────────
    log(f"Starting sliding-window inference (roi=96³, sw_batch={sw_batch_size}, overlap={overlap})...")
    roi_size = (96, 96, 96)

    try:
        with torch.inference_mode():
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = sliding_window_inference(
                    inputs=input_tensor,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode="gaussian",
                )
    except torch.cuda.OutOfMemoryError:
        log("⚠ GPU out of memory with sw_batch_size=2. Retrying with sw_batch_size=1...")
        torch.cuda.empty_cache()
        gc.collect()
        with torch.inference_mode():
            logits = sliding_window_inference(
                inputs=input_tensor,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=model,
                overlap=overlap,
                mode="gaussian",
            )

    log("Inference complete. Generating segmentation mask...")

    # ── 3. Post-processing ───────────────────────────────────────────────────
    # Argmax across class dimension → (1, D, H, W) → (D, H, W)
    seg_logits = logits[0]                        # (36, D, H, W)
    seg_mask = torch.argmax(seg_logits, dim=0)    # (D, H, W)
    seg_np = seg_mask.cpu().numpy().astype(np.uint8)   # (D, H, W)

    # ── 4. Save segmentation as NIfTI ────────────────────────────────────────
    log("Saving segmentation NIfTI...")

    # Use the original preprocessed affine from the transformed image
    meta = data.get("image_meta_dict", {})
    affine = meta.get("affine", np.eye(4))
    if hasattr(affine, "numpy"):
        affine = affine.numpy()

    # seg_np is (D, H, W) from RAS-resampled volume; nibabel wants (x,y,z)=(W,H,D)? 
    # Keep as (D, H, W) since MONAI returns it in that order after Orientation RAS
    seg_nii = nib.Nifti1Image(seg_np.transpose(2, 1, 0), affine)
    seg_nii.header.set_qform(affine, code=1)
    seg_nii.header.set_sform(affine, code=1)

    os.makedirs(os.path.dirname(output_seg_path), exist_ok=True)
    nib.save(seg_nii, output_seg_path)
    log(f"Segmentation saved to {output_seg_path} ✓")

    # ── 5. Memory cleanup ─────────────────────────────────────────────────────
    del logits, input_tensor, image_tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ── 6. Analyze segmentation ──────────────────────────────────────────────
    result_info = analyze_segmentation(seg_np)
    result_info["preprocessed_shape"] = list(seg_np.shape)

    return seg_np, result_info


def analyze_segmentation(seg_array: np.ndarray) -> Dict[str, Any]:
    """
    Analyze a segmentation mask and return presence info for all 35 classes.
    seg_array: (D, H, W) or (x, y, z) uint8 array with label values 0–35.
    """
    unique_labels = np.unique(seg_array).tolist()
    result = {
        "detected_labels": unique_labels,
        "structures": {},
        "detected_teeth": [],
        "teeth_count": 0,
    }

    min_voxels = 50  # threshold to consider a label "detected"

    for label_idx, label_name in LABEL_MAP.items():
        if label_idx == 0:
            continue
        count = int(np.sum(seg_array == label_idx))
        detected = count >= min_voxels
        result["structures"][label_idx] = {
            "name": label_name,
            "detected": detected,
            "voxel_count": count,
        }
        if detected and label_idx in TOOTH_FDI_MAP:
            result["detected_teeth"].append(TOOTH_FDI_MAP[label_idx])

    result["teeth_count"] = len(result["detected_teeth"])
    return result


def save_result_json(
    output_json_path: str,
    case_name: str,
    num_dicom_slices: int,
    volume_shape: list,
    voxel_spacing: list,
    device_str: str,
    checkpoint_name: str,
    seg_info: Dict[str, Any],
):
    """Save a non-PII results JSON file."""
    result = {
        "case": case_name,
        "input_type": "DICOM",
        "num_dicom_slices": num_dicom_slices,
        "volume_shape": volume_shape,
        "voxel_spacing_mm": voxel_spacing,
        "model": "OralSeg",
        "checkpoint": checkpoint_name,
        "device": device_str,
        "classes": 35,
        "detected_teeth": seg_info.get("detected_teeth", []),
        "teeth_detected_count": seg_info.get("teeth_count", 0),
        "structures_detected": {
            str(k): v["detected"]
            for k, v in seg_info.get("structures", {}).items()
        },
    }
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(result, f, indent=2, default=lambda x: bool(x) if isinstance(x, (np.bool_, bool)) else (int(x) if isinstance(x, (np.integer, int)) else (float(x) if isinstance(x, (np.floating, float)) else str(x))))

    return result
