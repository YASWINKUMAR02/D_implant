"""
dentalsegmentator_inference.py
==============================
Inference module for DentalSegmentator (nnU-Net v2, Dataset 112).

This module is completely independent from oralseg_inference.py.
Do NOT mix OralSeg and DentalSegmentator code.

DentalSegmentator label map (6 classes incl. background):
    0 = Background
    1 = Upper Skull / Maxilla
    2 = Mandible
    3 = Upper Teeth        ← grouped (NOT individual FDI teeth)
    4 = Lower Teeth        ← grouped (NOT individual FDI teeth)
    5 = Mandibular Canal

IMPORTANT: Labels 3 and 4 are aggregated tooth regions.
DentalSegmentator does NOT produce 32 individually-numbered FDI teeth.
Any downstream module that requires individual FDI tooth IDs (e.g. implant
planning, diagnocat_report, implant_suggester_3d) must check
seg_info["has_individual_fdi"] before use.

Architecture: nnU-Net v2, 3d_fullres configuration
Dataset: Dataset112_DentalSegmentator_v100
Trained on: 470 CBCT cases
Checkpoint: checkpoint_final.pth (fold 0)
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DentalSegmentator Label Map
# ─────────────────────────────────────────────────────────────────────────────
DS_LABEL_MAP: Dict[int, str] = {
    0: "Background",
    1: "Upper Skull / Maxilla",
    2: "Mandible",
    3: "Upper Teeth",
    4: "Lower Teeth",
    5: "Mandibular Canal",
}

# RGB colours (0–1 float) for 3D visualisation
DS_LABEL_COLORS: Dict[int, Tuple[float, float, float]] = {
    1: (0.85, 0.75, 0.60),   # Upper Skull — warm tan (bone)
    2: (0.40, 0.75, 0.45),   # Mandible — muted green
    3: (0.30, 0.60, 0.90),   # Upper Teeth — blue
    4: (0.90, 0.55, 0.25),   # Lower Teeth — orange
    5: (0.88, 0.25, 0.25),   # Mandibular Canal — red
}

# ─────────────────────────────────────────────────────────────────────────────
# Model Path Resolution
# ─────────────────────────────────────────────────────────────────────────────
# Project root is two directories above this file (utils/dentalsegmentator_inference.py)
_HERE: Path = Path(__file__).resolve().parent.parent


def get_model_dir() -> Path:
    """
    Resolve the path to the nnUNetTrainer__nnUNetPlans__3d_fullres directory.

    Priority:
      1. DENTALSEGMENTATOR_MODEL_DIR environment variable (full path to the
         nnUNetTrainer__nnUNetPlans__3d_fullres folder)
      2. Project-relative default:
         <project_root>/Dataset112_DentalSegmentator_v100/
           Dataset112_DentalSegmentator_v100/
             nnUNetTrainer__nnUNetPlans__3d_fullres/
    """
    env_override = os.environ.get("DENTALSEGMENTATOR_MODEL_DIR")
    if env_override:
        return Path(env_override)

    return (
        _HERE
        / "Dataset112_DentalSegmentator_v100"
        / "Dataset112_DentalSegmentator_v100"
        / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    )


def validate_model_files() -> Tuple[bool, str]:
    """
    Verify that all required nnU-Net model files are present on disk.

    Returns:
        (True, "")              if all files are present
        (False, error_message)  if any file is missing
    """
    model_dir = get_model_dir()
    required = {
        "plans.json":               model_dir / "plans.json",
        "dataset.json":             model_dir / "dataset.json",
        "dataset_fingerprint.json": model_dir / "dataset_fingerprint.json",
        "checkpoint_final.pth":     model_dir / "fold_0" / "checkpoint_final.pth",
    }
    for name, path in required.items():
        if not path.exists():
            return False, (
                f"Missing DentalSegmentator model file: {name}\n"
                f"Expected at: {path}\n"
                f"Model dir resolved to: {model_dir}\n"
                "Set DENTALSEGMENTATOR_MODEL_DIR env var to override."
            )
    return True, ""


def is_model_available() -> bool:
    """Quick check — True if all model files are present."""
    ok, _ = validate_model_files()
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Main Inference Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def run_ds_inference(
    nifti_path: str,
    output_seg_path: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run DentalSegmentator (nnU-Net v2) inference on a NIfTI volume.

    Args:
        nifti_path:        Path to input .nii or .nii.gz CBCT volume.
        output_seg_path:   Where to save the output segmentation (.nii.gz).
        progress_callback: Optional callable(message: str) for UI progress.

    Returns:
        seg_array  — uint8 numpy array (x, y, z) with labels 0–5.
        seg_info   — dict with model="dentalsegmentator" and detection flags.

    Raises:
        FileNotFoundError  — model checkpoint or input file missing.
        ImportError        — nnunetv2 package not installed.
        RuntimeError       — CUDA OOM, inference failure, or malformed output.
        ValueError         — invalid NIfTI (too small, empty intensity range).
    """
    import nibabel as nib
    import torch

    def log(msg: str) -> None:
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    # ── 1. Validate model files ───────────────────────────────────────────────
    ok, err_msg = validate_model_files()
    if not ok:
        raise FileNotFoundError(err_msg)

    model_dir = get_model_dir()

    # ── 2. Validate input NIfTI ───────────────────────────────────────────────
    nifti_path = str(nifti_path)
    if not os.path.exists(nifti_path):
        raise FileNotFoundError(f"Input NIfTI not found: {nifti_path}")

    try:
        probe = nib.load(nifti_path)
        probe_shape = probe.shape
        if any(d < 32 for d in probe_shape[:3]):
            raise ValueError(
                f"Input volume too small for nnU-Net inference: shape={probe_shape}. "
                "Minimum 32 voxels in each dimension required."
            )
        probe_data = probe.get_fdata(dtype=np.float32)
        if probe_data.max() <= probe_data.min():
            raise ValueError("Input NIfTI has zero intensity range — may be empty or corrupt.")
        del probe_data
    except Exception as e:
        raise ValueError(f"Invalid input NIfTI ({nifti_path}): {e}") from e

    # ── 3. Select device ─────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
        device_name = torch.cuda.get_device_name(0)
        log(f"Using GPU: {device_name}")
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        log(
            "⚠ CUDA unavailable — running DentalSegmentator on CPU. "
            "Inference will be very slow (30–90 min). GPU is strongly recommended."
        )

    # ── 4. Import nnunetv2 ────────────────────────────────────────────────────
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError as e:
        raise ImportError(
            f"nnunetv2 is not importable: {e}\n"
            "Install with: pip install nnunetv2\n"
            "Or install the specific version: pip install nnunetv2==2.8.1"
        ) from e

    # Free any leftover GPU / host memory before starting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 5. Initialise predictor ───────────────────────────────────────────────
    log("Initializing nnU-Net predictor...")
    # For memory efficiency on laptop GPUs (e.g. RTX 2050 4GB) and Windows:
    #   - perform_everything_on_device=False keeps prediction buffer on host RAM
    #   - use_mirroring=False avoids 8x redundant test-time augmentations (much faster & lower memory)
    #   - predict_from_files_sequential avoids multiprocessing spawn memory duplication on Windows
    predictor = nnUNetPredictor(
        tile_step_size=0.5,         # 50 % overlap between tiles
        use_gaussian=True,          # Gaussian weighting at tile edges
        use_mirroring=False,        # Disable TTA mirroring for speed and memory stability
        perform_everything_on_device=False,  # Essential for 4GB VRAM & large CBCTs
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,           # suppress tqdm in Streamlit
    )

    log(f"Loading DentalSegmentator checkpoint from:\n  {model_dir}")
    try:
        predictor.initialize_from_trained_model_folder(
            str(model_dir),
            use_folds=(0,),
            checkpoint_name="checkpoint_final.pth",
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load DentalSegmentator checkpoint: {e}\n"
            f"Model dir: {model_dir}"
        ) from e

    log("DentalSegmentator model loaded ✓")

    # ── 6. Prepare temp I/O directories ──────────────────────────────────────
    # nnU-Net predict_from_files reads/writes .nii.gz files via temp dirs.
    tmpdir = tempfile.mkdtemp(prefix="ds_infer_")
    seg_data: Optional[np.ndarray] = None

    try:
        tmp_in_path = str(Path(tmpdir) / "input_0000.nii.gz")
        tmp_out_dir = str(Path(tmpdir) / "output")
        os.makedirs(tmp_out_dir, exist_ok=True)

        # Copy / convert input to .nii.gz (nnU-Net requires _0000 suffix for channel)
        in_p = Path(nifti_path)
        if nifti_path.endswith(".nii.gz"):
            shutil.copy2(nifti_path, tmp_in_path)
        else:
            # .nii → .nii.gz
            img = nib.load(nifti_path)
            nib.save(img, tmp_in_path)

        # ── 7. Run inference ──────────────────────────────────────────────────
        log(
            "Starting nnU-Net 3D full-res inference "
            "(patch [128×160×112], tile_step=0.5, sequential mode)..."
        )
        try:
            predictor.predict_from_files_sequential(
                list_of_lists_or_source_folder=[[tmp_in_path]],
                output_folder_or_list_of_truncated_output_files=[
                    os.path.join(tmp_out_dir, "seg")
                ],
                save_probabilities=False,
                overwrite=True,
                folder_with_segs_from_prev_stage=None,
            )
        except torch.cuda.OutOfMemoryError as oom:
            raise RuntimeError(
                "GPU out of memory during DentalSegmentator inference.\n"
                "Suggestions:\n"
                "  • Close other GPU applications and retry\n"
                "  • Restart Streamlit to free VRAM\n"
                f"Original error: {oom}"
            ) from oom

        log("nnU-Net inference complete ✓")

        # ── 8. Read output segmentation ────────────────────────────────────────
        out_nii_path = os.path.join(tmp_out_dir, "seg.nii.gz")
        if not os.path.exists(out_nii_path):
            raise RuntimeError(
                f"nnU-Net did not produce expected output at: {out_nii_path}\n"
                "This usually means inference failed silently. "
                "Check the Python console for nnU-Net error messages."
            )

        seg_img = nib.load(out_nii_path)
        seg_data = seg_img.get_fdata(dtype=np.float32).astype(np.uint8)
        log(
            f"Segmentation shape: {seg_data.shape}, "
            f"unique labels: {np.unique(seg_data).tolist()}"
        )

        # Validate labels
        unexpected = set(np.unique(seg_data).tolist()) - {0, 1, 2, 3, 4, 5}
        if unexpected:
            logger.warning(
                f"DentalSegmentator produced unexpected label values: {unexpected}. "
                "Output may be from an incompatible model version."
            )

        # ── 9. Save to requested output path ───────────────────────────────────
        os.makedirs(os.path.dirname(os.path.abspath(output_seg_path)), exist_ok=True)

        # Prefer to use the seg_img's own affine (nnU-Net preserves input geometry)
        out_affine = seg_img.affine
        out_header = seg_img.header

        out_nii = nib.Nifti1Image(seg_data, out_affine, out_header)
        out_nii.header.set_qform(out_affine, code=1)
        out_nii.header.set_sform(out_affine, code=1)
        nib.save(out_nii, output_seg_path)
        log(f"Segmentation saved to: {output_seg_path} ✓")

    finally:
        # Always clean up temp dir and free GPU
        shutil.rmtree(tmpdir, ignore_errors=True)
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ── 10. Build seg_info dict ────────────────────────────────────────────────
    seg_info = _analyze_ds_segmentation(seg_data, device_name)
    return seg_data, seg_info


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation Analysis
# ─────────────────────────────────────────────────────────────────────────────
def _analyze_ds_segmentation(
    seg_array: np.ndarray,
    device_name: str = "unknown",
) -> Dict[str, Any]:
    """
    Build a seg_info dict from a DentalSegmentator output array.

    IMPORTANT: Unlike OralSeg, detected_teeth is always [] because
    DentalSegmentator groups all upper/lower teeth into labels 3/4.
    Downstream modules must check has_individual_fdi before using tooth IDs.
    """
    unique_labels = [int(v) for v in np.unique(seg_array).tolist()]
    structures: Dict[int, Dict[str, Any]] = {}
    min_voxels = 50  # minimum voxel count to count as "detected"

    for label_idx, label_name in DS_LABEL_MAP.items():
        if label_idx == 0:
            continue
        count = int(np.sum(seg_array == label_idx))
        structures[label_idx] = {
            "name": label_name,
            "detected": count >= min_voxels,
            "voxel_count": count,
        }

    return {
        # ── Model identifier ──────────────────────────────────────────────
        "model": "dentalsegmentator",
        "device_name": device_name,
        # ── Detection data ────────────────────────────────────────────────
        "detected_labels": unique_labels,
        "structures": structures,
        # ── Tooth compatibility flags ─────────────────────────────────────
        # DentalSegmentator does NOT produce individual FDI-numbered teeth.
        # These fields are kept for interface compatibility with OralSeg
        # result dicts but are intentionally empty/False.
        "detected_teeth": [],        # always empty — no per-tooth FDI IDs
        "teeth_count": 0,            # no individual tooth count
        "has_individual_fdi": False, # ← gate flag for downstream modules
        # ── DS-specific detected structures ──────────────────────────────
        "upper_skull_detected":      structures.get(1, {}).get("detected", False),
        "mandible_detected":         structures.get(2, {}).get("detected", False),
        "upper_teeth_detected":      structures.get(3, {}).get("detected", False),
        "lower_teeth_detected":      structures.get(4, {}).get("detected", False),
        "mandibular_canal_detected": structures.get(5, {}).get("detected", False),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Result JSON helper
# ─────────────────────────────────────────────────────────────────────────────
def save_ds_result_json(
    output_json_path: str,
    case_name: str,
    num_dicom_slices: int,
    volume_shape: list,
    voxel_spacing: list,
    device_str: str,
    seg_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Save a non-PII results JSON for a DentalSegmentator segmentation."""
    result = {
        "case": case_name,
        "model": "DentalSegmentator",
        "dataset": "Dataset112_DentalSegmentator_v100",
        "architecture": "nnU-Net 3d_fullres",
        "fold": 0,
        "device": device_str,
        "classes": 5,  # excluding background
        "label_map": DS_LABEL_MAP,
        "has_individual_fdi_teeth": False,
        "num_dicom_slices": num_dicom_slices,
        "volume_shape": volume_shape,
        "voxel_spacing_mm": voxel_spacing,
        "structures_detected": {
            str(k): v["detected"]
            for k, v in seg_info.get("structures", {}).items()
        },
        "note": (
            "DentalSegmentator provides grouped upper/lower tooth labels. "
            "Individual FDI tooth IDs (11-18, 21-28, 31-38, 41-48) are NOT available. "
            "Use OralSeg for 32-tooth individual FDI segmentation."
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result
