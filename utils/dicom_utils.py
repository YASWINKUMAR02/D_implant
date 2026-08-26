"""
dicom_utils.py
Handles ZIP extraction, DICOM series discovery, validation, and metadata extraction.
All patient-identifying information is explicitly excluded from returned metadata.
"""

import os
import sys
import site
from pathlib import Path

user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.insert(0, user_site)

roaming_site = os.path.expandvars(r"%APPDATA%\Python\Python311\site-packages")
if os.path.exists(roaming_site) and roaming_site not in sys.path:
    sys.path.insert(0, roaming_site)

import zipfile
import tempfile
import shutil
from typing import Tuple, Dict, Any, List, Optional

import pydicom
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# DICOM tags that must NEVER be returned (PII / PHI filter)
# ─────────────────────────────────────────────────────────────────────────────
_BLOCKED_TAGS = {
    (0x0010, 0x0010),  # PatientName
    (0x0010, 0x0020),  # PatientID
    (0x0010, 0x0030),  # PatientBirthDate
    (0x0010, 0x0040),  # PatientSex
    (0x0010, 0x1000),  # OtherPatientIDs
    (0x0008, 0x0090),  # ReferringPhysicianName
    (0x0008, 0x1048),  # PhysicianOfRecord
    (0x0032, 0x1032),  # RequestingPhysician
    (0x0040, 0xA123),  # PersonName
    (0x0008, 0x0080),  # InstitutionName
    (0x0008, 0x0081),  # InstitutionAddress
}


def extract_zip(zip_bytes: bytes, extract_dir: str) -> str:
    """
    Extract a ZIP archive from bytes into extract_dir.
    Returns the path to the extraction directory.
    Raises ValueError for invalid ZIP.
    """
    if not zipfile.is_zipfile(__import__('io').BytesIO(zip_bytes)):
        raise ValueError("Uploaded file is not a valid ZIP archive.")

    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(__import__('io').BytesIO(zip_bytes), 'r') as zf:
        # Security: strip absolute paths and block path traversal
        for member in zf.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                continue
            zf.extract(member, extract_dir)

    return extract_dir


def find_dicom_files(directory: str) -> List[str]:
    """
    Recursively find all DICOM files in a directory.
    A file is considered DICOM if it has a .dcm extension OR passes pydicom validation.
    Returns sorted list of absolute paths.
    """
    dicom_files = []
    for root, dirs, files in os.walk(directory):
        for fname in files:
            fpath = os.path.join(root, fname)
            # Check extension first (fast)
            if fname.lower().endswith('.dcm'):
                dicom_files.append(fpath)
            else:
                # Try reading with pydicom (handles files without .dcm extension)
                try:
                    ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=False)
                    if hasattr(ds, 'Modality'):
                        dicom_files.append(fpath)
                except Exception:
                    pass

    return sorted(dicom_files)


def group_by_series(dicom_files: List[str]) -> Dict[str, List[str]]:
    """
    Group DICOM files by SeriesInstanceUID.
    Returns dict mapping series UID -> list of file paths.
    """
    series_map: Dict[str, List[str]] = {}
    for fpath in dicom_files:
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            uid = str(getattr(ds, 'SeriesInstanceUID', 'unknown'))
            series_map.setdefault(uid, []).append(fpath)
        except Exception:
            series_map.setdefault('unknown', []).append(fpath)

    return series_map


def validate_series(series_files: List[str]) -> Dict[str, Any]:
    """
    Validate a DICOM series and return non-PII metadata.
    Raises ValueError with a descriptive message on failure.

    Returns dict with:
      - num_slices: int
      - modality: str
      - pixel_spacing: list[float] or None
      - slice_thickness: float or None
      - rows: int
      - columns: int
      - bits_allocated: int
    """
    if not series_files:
        raise ValueError("No DICOM files found in the uploaded archive.")

    # Read the first slice for metadata
    try:
        ds = pydicom.dcmread(series_files[0], stop_before_pixels=True)
    except Exception as e:
        raise ValueError(f"Cannot read DICOM file: {e}")

    modality = str(getattr(ds, 'Modality', 'UNKNOWN'))
    rows = int(getattr(ds, 'Rows', 0))
    cols = int(getattr(ds, 'Columns', 0))
    bits = int(getattr(ds, 'BitsAllocated', 16))

    ps = getattr(ds, 'PixelSpacing', None)
    pixel_spacing = [float(ps[0]), float(ps[1])] if ps and len(ps) >= 2 else None

    st = getattr(ds, 'SliceThickness', None)
    slice_thickness = float(st) if st is not None else None

    return {
        "num_slices": len(series_files),
        "modality": modality,
        "pixel_spacing": pixel_spacing,
        "slice_thickness": slice_thickness,
        "rows": rows,
        "columns": cols,
        "bits_allocated": bits,
    }


def inspect_zip(zip_bytes: bytes, extract_dir: str) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """
    High-level function: extract ZIP, find DICOM files, group by series, validate.

    Returns:
        series_map: dict of {series_uid: [file_paths]}
        summary:    non-PII metadata dict

    Raises:
        ValueError on any validation failure.
    """
    extract_zip(zip_bytes, extract_dir)
    dicom_files = find_dicom_files(extract_dir)

    if not dicom_files:
        raise ValueError("No DICOM files (.dcm) found in the uploaded ZIP archive.")

    series_map = group_by_series(dicom_files)

    # Select the largest series automatically
    best_uid = max(series_map, key=lambda k: len(series_map[k]))
    best_files = series_map[best_uid]

    meta = validate_series(best_files)
    meta["series_count"] = len(series_map)
    meta["selected_series_uid"] = best_uid
    meta["all_dicom_files"] = len(dicom_files)

    if len(series_map) > 1:
        meta["multi_series_warning"] = (
            f"ZIP contains {len(series_map)} DICOM series. "
            f"Using the largest series ({len(best_files)} slices)."
        )

    return series_map, meta


def get_series_files(series_map: Dict[str, List[str]], series_uid: Optional[str] = None) -> List[str]:
    """Return files for a specific series UID, or the largest series."""
    if series_uid and series_uid in series_map:
        return series_map[series_uid]
    return series_map[max(series_map, key=lambda k: len(series_map[k]))]
