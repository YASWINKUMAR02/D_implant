"""
nifti_utils.py
Converts a DICOM series directory to NIfTI (.nii.gz) using SimpleITK.
Preserves voxel spacing, orientation, and affine information.
"""

import os
from typing import List, Dict, Any, Tuple

import numpy as np
import nibabel as nib
import SimpleITK as sitk


def dicom_series_to_nifti(
    dicom_files: List[str],
    output_path: str,
    progress_callback=None
) -> Tuple[str, Dict[str, Any]]:
    """
    Convert a list of DICOM slice files to a NIfTI volume.

    Args:
        dicom_files: sorted list of DICOM file paths belonging to one series
        output_path: path to save the output .nii.gz file
        progress_callback: optional callable(message: str) for progress reporting

    Returns:
        (output_path, volume_info) where volume_info is a non-PII metadata dict

    Raises:
        RuntimeError on conversion failure
    """
    def log(msg):
        if progress_callback:
            progress_callback(msg)

    # ── 1. Use SimpleITK ImageSeriesReader ──────────────────────────────────
    log("Setting up DICOM series reader...")
    reader = sitk.ImageSeriesReader()

    # Get the directory from the first file
    dicom_dir = os.path.dirname(dicom_files[0])

    # Get all series IDs in this directory
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)

    if series_ids:
        # Use the series containing the most files
        best_series_id = None
        max_files = 0
        for sid in series_ids:
            fnames = reader.GetGDCMSeriesFileNames(dicom_dir, sid)
            if len(fnames) > max_files:
                max_files = len(fnames)
                best_series_id = sid

        if best_series_id:
            series_file_names = reader.GetGDCMSeriesFileNames(dicom_dir, best_series_id)
            reader.SetFileNames(series_file_names)
        else:
            reader.SetFileNames(dicom_files)
    else:
        # Fall back: use the provided file list directly
        reader.SetFileNames(dicom_files)

    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    log("Reading DICOM series...")
    try:
        image = reader.Execute()
    except Exception as e:
        raise RuntimeError(f"SimpleITK failed to read DICOM series: {e}")

    # ── 2. Log volume geometry ───────────────────────────────────────────────
    spacing = image.GetSpacing()          # (x, y, z) in mm
    size = image.GetSize()                # (x, y, z) voxels
    origin = image.GetOrigin()
    direction = image.GetDirection()

    log(f"Volume size: {size[0]}×{size[1]}×{size[2]} voxels")
    log(f"Voxel spacing: {spacing[0]:.3f}×{spacing[1]:.3f}×{spacing[2]:.3f} mm")

    # ── 3. Convert to numpy and validate ────────────────────────────────────
    array = sitk.GetArrayFromImage(image)   # shape: (z, y, x)
    arr_min, arr_max = int(array.min()), int(array.max())

    log(f"Intensity range: [{arr_min}, {arr_max}] HU")

    if arr_max <= arr_min:
        raise RuntimeError("Volume has zero intensity range — likely an empty or corrupt DICOM series.")

    # ── 4. Save as NIfTI via nibabel ─────────────────────────────────────────
    log("Building NIfTI affine matrix...")

    # Build affine from SimpleITK origin, spacing, direction
    dir_arr = np.array(direction).reshape(3, 3)
    sp_arr = np.diag(spacing)
    rot = dir_arr @ sp_arr

    # SimpleITK uses LPS convention; nibabel uses RAS by default
    # Flip X and Y axes for LPS→RAS
    flip = np.diag([-1, -1, 1])
    rot_ras = flip @ rot
    origin_ras = flip @ np.array(origin)

    affine = np.eye(4)
    affine[:3, :3] = rot_ras
    affine[:3, 3] = origin_ras

    # nibabel expects (x, y, z) so transpose from (z, y, x)
    volume_xyz = array.transpose(2, 1, 0).astype(np.int16)

    nii_img = nib.Nifti1Image(volume_xyz, affine)

    # Set qform/sform
    nii_img.header.set_qform(affine, code=1)
    nii_img.header.set_sform(affine, code=1)
    nii_img.header['pixdim'][1:4] = spacing

    log(f"Saving NIfTI to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(nii_img, output_path)

    volume_info = {
        "volume_shape": list(volume_xyz.shape),     # (x, y, z)
        "voxel_spacing_mm": list(spacing),           # (x, y, z)
        "intensity_range": [arr_min, arr_max],
        "num_slices": size[2],
        "nifti_path": output_path,
    }

    log("DICOM → NIfTI conversion complete ✓")
    return output_path, volume_info


def load_nifti_volume(nifti_path: str) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Load a NIfTI file and return (data_array, affine, header).
    data_array shape: (x, y, z)
    """
    img = nib.load(nifti_path)
    return img.get_fdata(dtype=np.float32), img.affine, img.header


def validate_nifti_for_inference(nifti_path: str) -> Dict[str, Any]:
    """
    Load and validate a NIfTI volume for OralSeg inference.
    Returns volume info dict.
    """
    img = nib.load(nifti_path)
    data = img.get_fdata(dtype=np.float32)
    spacing = img.header.get_zooms()[:3]

    info = {
        "shape": list(data.shape),
        "spacing_mm": [float(s) for s in spacing],
        "intensity_min": float(data.min()),
        "intensity_max": float(data.max()),
        "dtype": str(data.dtype),
    }

    # Basic sanity checks
    if any(d < 32 for d in data.shape):
        raise ValueError(
            f"Volume too small for inference: shape={data.shape}. "
            "Minimum 32 voxels in each dimension required."
        )

    if data.max() <= data.min():
        raise ValueError("NIfTI volume has zero intensity range — the image may be empty.")

    return info
