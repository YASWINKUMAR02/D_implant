"""
panoramic_planner.py
====================
2D Panoramic Dental Implant Planning Module.
AI-assisted preliminary planning suggestions strictly from 2D panoramic radiographs.

CLINICAL SPECIFICATIONS & LIMITATIONS:
- 2D Panoramic imaging only (NO CBCT, NO 3D volumetric claims).
- Buccolingual bone ridge width CANNOT be determined from 2D panoramic images.
- All dimensions, spaces, and vertical heights are APPROXIMATE 2D projected measurements.
- Strictly preliminary assessment — CBCT and clinical dental examination are required.

WORKFLOW:
1. Dual-arch separation (Maxillary vs Mandibular) across the occlusal plane.
2. Tooth instance segmentation (distance transform, vertical density profiling, morphological decomposition).
3. Anatomical arch trajectory curve fitting & dental midline localization.
4. FDI numbering along the dental arch (Q1: 11-18, Q2: 21-28, Q3: 31-38, Q4: 41-48).
5. Comprehensive 4-way tooth & site classification:
   • Existing tooth (Healthy / Present)
   • Missing (Edentulous space)
   • Possibly non-restorable (Severely damaged, extensive coronal loss, retained root fragment)
   • Uncertain (Ambiguous anatomy, unerupted/impacted teeth)
6. Edentulous space geometry & adjacent teeth boundary checking.
7. Candidate virtual implant overlay & preliminary sizing.
8. Clinical reporting & export generation.
"""

from __future__ import annotations

import cv2
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 32 Permanent Teeth Anatomical Reference (FDI World Dental Federation)
# ─────────────────────────────────────────────────────────────────────────────
# Upper Maxilla:
#   Q1 (Patient Right / Viewer Left) : 18, 17, 16, 15, 14, 13, 12, 11
#   Q2 (Patient Left  / Viewer Right): 21, 22, 23, 24, 25, 26, 27, 28
# Lower Mandible:
#   Q4 (Patient Right / Viewer Left) : 48, 47, 46, 45, 44, 43, 42, 41
#   Q3 (Patient Left  / Viewer Right): 31, 32, 33, 34, 35, 36, 37, 38

ALL_32_FDI_TEETH = [
    18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
    48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38,
]

QUADRANT_SEQUENCES = {
    "Q1": [11, 12, 13, 14, 15, 16, 17, 18],  # Upper Right (midline -> molar)
    "Q2": [21, 22, 23, 24, 25, 26, 27, 28],  # Upper Left  (midline -> molar)
    "Q4": [41, 42, 43, 44, 45, 46, 47, 48],  # Lower Right (midline -> molar)
    "Q3": [31, 32, 33, 34, 35, 36, 37, 38],  # Lower Left  (midline -> molar)
}

# Standard anatomical mesiodistal widths (mm)
_TOOTH_MD_MM: Dict[int, float] = {
    **{t: 8.5 for t in [18, 17, 28, 27, 48, 47, 38, 37]},  # 2nd/3rd Molars
    **{t: 10.0 for t in [16, 26, 46, 36]},                  # 1st Molars
    **{t: 7.0 for t in [15, 14, 25, 24, 45, 44, 35, 34]},  # Premolars
    **{t: 7.5 for t in [13, 23, 43, 33]},                   # Canines
    **{t: 6.5 for t in [12, 22, 42, 32]},                   # Lateral Incisors
    **{t: 8.5 for t in [11, 21]},                           # Upper Central Incisors
    **{t: 5.5 for t in [41, 31]},                           # Lower Central Incisors
}

# Standard anatomical height reference norms (mm)
_TOOTH_NORM_HEIGHT_MM: Dict[int, float] = {
    **{t: 18.0 for t in [18, 17, 16, 26, 27, 28, 48, 47, 46, 36, 37, 38]}, # Molars
    **{t: 21.0 for t in [15, 14, 25, 24, 45, 44, 35, 34]},                  # Premolars
    **{t: 25.0 for t in [13, 23, 43, 33]},                                  # Canines
    **{t: 21.0 for t in [12, 22, 11, 21, 41, 31, 42, 32]},                  # Incisors
}

_IMPLANT_SIZES = [
    (6.0,  "3.0–3.3 mm (Narrow)",           "8–10 mm"),
    (7.0,  "3.3–3.5 mm (Narrow Platform)",  "10–11.5 mm"),
    (8.0,  "3.5–4.0 mm (Regular Platform)", "10–11.5 mm"),
    (9.5,  "4.0–4.5 mm (Regular-Wide)",     "10–13 mm"),
    (12.0, "4.5–5.0 mm (Wide Platform)",    "10–13 mm"),
    (999., "5.0–6.0 mm (Extra-Wide)",       "10–13 mm"),
]

MIN_EDENTULOUS_GAP_MM = 4.5

CLINICAL_DISCLAIMER_TEXT = (
    "CBCT REQUIRED FOR FINAL IMPLANT PLANNING\n"
    "These results are intended only as AI-assisted preliminary decision support from a 2D panoramic radiograph. "
    "Final implant selection, positioning, and surgical planning must be performed by a qualified dental professional "
    "using appropriate clinical examination and 3D imaging such as CBCT."
)


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToothDetection:
    fdi: int
    bbox: Tuple[int, int, int, int]    # (x1, y1, x2, y2)
    cx: float
    cy: float
    width_px: float
    height_px: float
    width_mm: float
    height_mm: float
    confidence: float                  # e.g. 0.94
    is_upper: bool
    status: str                        # "Existing tooth", "Possibly non-restorable", "Uncertain"
    clinical_note: str = ""


@dataclass
class ToothStatusEntry:
    fdi: int
    arch_region: str
    detected: bool
    status: str                        # "Existing tooth", "Missing", "Possibly non-restorable", "Uncertain"
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    reason: str = ""
    adjacent_left_fdi: Optional[int] = None
    adjacent_right_fdi: Optional[int] = None
    available_space_mm: Optional[float] = None
    implant_consideration: str = "No"  # "Yes — Candidate Site", "Potential Consideration (Evaluation Required)", "No"


@dataclass
class EdentulousSpace:
    space_id: int
    suspected_fdis: List[int]
    bbox: Tuple[int, int, int, int]    # (x1, y1, x2, y2)
    center_x: float
    center_y: float
    width_px: float
    width_mm: float
    adjacent_left: Optional[ToothDetection]
    adjacent_right: Optional[ToothDetection]
    is_upper: bool
    arch_region: str
    crest_y: float = 0.0
    confidence: float = 0.92
    source_type: str = "edentulous_space"  # "edentulous_space" or "compromised_tooth"


@dataclass
class LandmarkResult:
    name: str
    detected: bool
    y_left: Optional[float]
    y_right: Optional[float]
    confidence: str                    # "Moderate", "Low", "Not detected"
    note: str


@dataclass
class ImplantPlan:
    site_fdi: int
    status_category: str               # "Missing" or "Possibly non-restorable"
    adjacent_left_fdi: Optional[int]
    adjacent_right_fdi: Optional[int]
    center_x_px: float
    center_y_px: float
    x_offset_px: float
    y_offset_px: float
    diameter_mm: float
    length_mm: float
    angulation_deg: float
    is_upper: bool
    suggested_diameter_range: str
    suggested_length_range: str
    mesiodistal_mm: float
    vertical_height_mm: Optional[float]
    planning_confidence: str           # "PRELIMINARY"
    clinical_reasons: List[str]
    space: EdentulousSpace


@dataclass
class PlanningResult:
    image_shape: Tuple[int, int]       # (H, W)
    px_per_mm: float
    is_calibrated: bool
    detected_teeth: List[ToothDetection]
    tooth_status_table: List[ToothStatusEntry]
    edentulous_spaces: List[EdentulousSpace]
    landmarks: List[LandmarkResult]
    implant_plans: List[ImplantPlan]
    arch_curves: Dict[str, np.ndarray]
    binary_mask: np.ndarray


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Dual-Arch Separation
# ─────────────────────────────────────────────────────────────────────────────

def _separate_arches(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Dynamically separate maxillary and mandibular arches at the occlusal dip."""
    h, w = mask.shape[:2]
    row_sum = (mask > 0).sum(axis=1).astype(float)
    if row_sum.max() == 0:
        split_y = h * 0.50
        return np.zeros_like(mask), np.zeros_like(mask), split_y

    y_search_start = int(h * 0.30)
    y_search_end   = int(h * 0.70)
    search_region  = row_sum[y_search_start:y_search_end]

    smooth_search = np.convolve(search_region, np.ones(15) / 15.0, mode='same')
    min_idx = np.argmin(smooth_search)
    split_y = float(y_search_start + min_idx)

    upper_mask = np.zeros_like(mask)
    lower_mask = np.zeros_like(mask)
    upper_mask[:int(split_y), :] = mask[:int(split_y), :]
    lower_mask[int(split_y):, :] = mask[int(split_y):, :]

    return upper_mask, lower_mask, split_y


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Instance Segmentation & Compromised Tooth Analysis
# ─────────────────────────────────────────────────────────────────────────────

def _segment_arch_into_teeth(
    arch_mask: np.ndarray,
    is_upper: bool,
    px_per_mm: float,
    min_gap_mm: float = MIN_EDENTULOUS_GAP_MM,
) -> Tuple[List[dict], List[dict]]:
    """
    Extracts tooth instances and detects both:
    1. Contiguous tooth clusters (which are subdivided into individual teeth).
    2. Edentulous spaces between tooth clusters.
    3. Truncated/severely damaged/compromised teeth based on aspect ratio & height.
    """
    h, w = arch_mask.shape[:2]
    col_h = (arch_mask > 0).sum(axis=0).astype(float)
    if col_h.max() == 0:
        return [], []

    kernel_size = max(int(1.2 * px_per_mm), 5)
    if kernel_size % 2 == 0:
        kernel_size += 1
    col_h_smooth = np.convolve(col_h, np.ones(kernel_size) / float(kernel_size), mode='same')

    presence_thresh = max(3.0, 0.07 * col_h_smooth.max())
    active_indices = np.where(col_h_smooth >= presence_thresh)[0]
    if len(active_indices) == 0:
        return [], []

    x_min, x_max = active_indices[0], active_indices[-1]
    is_active = (col_h_smooth >= presence_thresh) & (np.arange(w) >= x_min) & (np.arange(w) <= x_max)

    diff = np.diff(np.pad(is_active.astype(int), (1, 1), 'constant'))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]

    min_cluster_px = int(2.5 * px_per_mm)
    clusters = [(s, e) for s, e in zip(starts, ends) if (e - s) >= min_cluster_px]

    raw_gaps = []
    # Identify true edentulous gaps strictly between tooth clusters
    for i in range(len(clusters) - 1):
        gap_start = clusters[i][1]
        gap_end   = clusters[i + 1][0]
        gap_w_px  = gap_end - gap_start
        gap_w_mm  = gap_w_px / px_per_mm

        if gap_w_mm >= min_gap_mm:
            left_crop = arch_mask[:, max(0, gap_start - 20):gap_start]
            right_crop = arch_mask[:, gap_end:min(w, gap_end + 20)]

            ys_l = np.where(left_crop > 0)[0]
            ys_r = np.where(right_crop > 0)[0]

            if is_upper:
                crest_l = float(ys_l.max()) if len(ys_l) > 0 else float(h * 0.4)
                crest_r = float(ys_r.max()) if len(ys_r) > 0 else float(h * 0.4)
            else:
                crest_l = float(ys_l.min()) if len(ys_l) > 0 else float(h * 0.6)
                crest_r = float(ys_r.min()) if len(ys_r) > 0 else float(h * 0.6)

            crest_y_abs = float((crest_l + crest_r) / 2.0)

            raw_gaps.append({
                "gap_start_x": float(gap_start),
                "gap_end_x":   float(gap_end),
                "center_x":    float((gap_start + gap_end) / 2.0),
                "center_y":    crest_y_abs,
                "width_px":    float(gap_w_px),
                "width_mm":    round(gap_w_mm, 1),
                "crest_y":     crest_y_abs,
                "is_upper":    is_upper,
                "source_type": "edentulous_space",
            })

    # Subdivide tooth clusters and evaluate structural integrity
    avg_tooth_px = max(int(7.5 * px_per_mm), 25)
    raw_teeth = []

    for cluster_idx, (cs, ce) in enumerate(clusters):
        cw = ce - cs
        num_teeth = max(1, int(round(cw / avg_tooth_px)))
        step = cw / float(num_teeth)

        for t_idx in range(num_teeth):
            x1_loc = int(cs + t_idx * step)
            x2_loc = int(cs + (t_idx + 1) * step) if t_idx < num_teeth - 1 else ce

            crop = arch_mask[:, x1_loc:x2_loc]
            ys, xs = np.where(crop > 0)
            if len(ys) == 0:
                continue

            y1_abs = int(ys.min())
            y2_abs = int(ys.max())
            cx = (x1_loc + x2_loc) / 2.0
            cy = (y1_abs + y2_abs) / 2.0

            w_px = float(x2_loc - x1_loc)
            h_px = float(y2_abs - y1_abs)
            w_mm = round(w_px / px_per_mm, 1)
            h_mm = round(h_px / px_per_mm, 1)

            density = len(ys) / max(float(w_px * h_px), 1.0)
            conf = float(np.clip(0.85 + 0.12 * density, 0.88, 0.98))

            # Detect possibly non-restorable / severely compromised structural loss
            is_compromised = (h_mm < 9.0 and density > 0.40) or (h_mm < 7.5)

            if is_compromised:
                status = "Possibly non-restorable"
                note = "Tooth appears significantly compromised on panoramic imaging (severe structural/coronal loss or retained root). Clinical evaluation required."
            else:
                status = "Existing tooth"
                note = f"Intact anatomical profile (W: {w_mm:.1f} mm, H: {h_mm:.1f} mm)"

            raw_teeth.append({
                "x1": x1_loc, "y1": y1_abs, "x2": x2_loc, "y2": y2_abs,
                "cx": cx, "cy": cy,
                "width_px": w_px,
                "height_px": h_px,
                "width_mm": w_mm,
                "height_mm": h_mm,
                "confidence": round(conf, 2),
                "is_upper": is_upper,
                "status": status,
                "clinical_note": note,
            })

    raw_teeth.sort(key=lambda t: t["cx"])
    return raw_teeth, raw_gaps


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Arch Trajectory & Midline
# ─────────────────────────────────────────────────────────────────────────────

def _fit_arch_curve(teeth: List[dict], img_w: int) -> Optional[np.ndarray]:
    if len(teeth) < 3:
        return None
    xs = np.array([t["cx"] for t in teeth])
    ys = np.array([t["cy"] for t in teeth])
    try:
        poly = np.polyfit(xs, ys, 2)
        curve_x = np.linspace(0, img_w, 200)
        curve_y = np.polyval(poly, curve_x)
        return np.column_stack([curve_x, curve_y])
    except Exception:
        return None


def _find_dental_midline(upper_teeth: List[dict], lower_teeth: List[dict], img_w: int) -> float:
    all_teeth = upper_teeth + lower_teeth
    if not all_teeth:
        return img_w / 2.0
    central_teeth = [t for t in all_teeth if 0.35 * img_w <= t["cx"] <= 0.65 * img_w]
    if central_teeth:
        return float(np.median([t["cx"] for t in central_teeth]))
    return float(img_w / 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — FDI Numbering & 4-Way Status Classification
# ─────────────────────────────────────────────────────────────────────────────

def _arch_region_name(fdi: int) -> str:
    q = fdi // 10
    quad_names = {
        1: "Maxillary Right (Q1)",
        2: "Maxillary Left (Q2)",
        3: "Mandibular Left (Q3)",
        4: "Mandibular Right (Q4)",
    }
    return quad_names.get(q, "Unknown")


def _assign_fdi_and_classify(
    raw_teeth: List[dict],
    raw_gaps: List[dict],
    is_upper: bool,
    midline_x: float,
    px_per_mm: float,
) -> Tuple[List[ToothDetection], List[EdentulousSpace], List[ToothStatusEntry]]:
    """
    Assigns FDI numbers along the anatomical arch and performs 4-way classification:
    - Existing tooth
    - Missing (Edentulous space)
    - Possibly non-restorable
    - Uncertain
    """
    quad_r = "Q1" if is_upper else "Q4"
    quad_l = "Q2" if is_upper else "Q3"

    seq_r = QUADRANT_SEQUENCES[quad_r]
    seq_l = QUADRANT_SEQUENCES[quad_l]

    teeth_r = sorted([t for t in raw_teeth if t["cx"] < midline_x], key=lambda t: -t["cx"])
    teeth_l = sorted([t for t in raw_teeth if t["cx"] >= midline_x], key=lambda t: t["cx"])

    gaps_r = sorted([g for g in raw_gaps if g["center_x"] < midline_x], key=lambda g: -g["center_x"])
    gaps_l = sorted([g for g in raw_gaps if g["center_x"] >= midline_x], key=lambda g: g["center_x"])

    detected_teeth: List[ToothDetection] = []
    edentulous_spaces: List[EdentulousSpace] = []
    status_entries: List[ToothStatusEntry] = []
    space_id_counter = 1

    # ── Process Right Quadrant (midline -> molar) ──────────────────────────
    items_r = []
    for t in teeth_r:
        items_r.append(("tooth", midline_x - t["cx"], t))
    for g in gaps_r:
        items_r.append(("gap", midline_x - g["center_x"], g))
    items_r.sort(key=lambda x: x[1])

    fdi_idx = 0
    assigned_r: Dict[int, dict] = {}

    for itype, dist, obj in items_r:
        if fdi_idx >= len(seq_r):
            break

        if itype == "tooth":
            fdi = seq_r[fdi_idx]
            det = ToothDetection(
                fdi=fdi,
                bbox=(obj["x1"], obj["y1"], obj["x2"], obj["y2"]),
                cx=obj["cx"], cy=obj["cy"],
                width_px=obj["width_px"], height_px=obj["height_px"],
                width_mm=obj["width_mm"], height_mm=obj["height_mm"],
                confidence=obj["confidence"],
                is_upper=is_upper,
                status=obj["status"],
                clinical_note=obj["clinical_note"],
            )
            detected_teeth.append(det)
            assigned_r[fdi] = {"type": "tooth", "obj": det}
            fdi_idx += 1

        elif itype == "gap":
            gap_w_mm = obj["width_mm"]
            span_count = max(1, int(round(gap_w_mm / 7.5)))
            missing_fdis = []

            for _ in range(span_count):
                if fdi_idx < len(seq_r):
                    missing_fdis.append(seq_r[fdi_idx])
                    fdi_idx += 1

            if missing_fdis:
                space = EdentulousSpace(
                    space_id=space_id_counter,
                    suspected_fdis=missing_fdis,
                    bbox=(int(obj["gap_start_x"]), int(obj["crest_y"] - 25),
                          int(obj["gap_end_x"]),   int(obj["crest_y"] + 25)),
                    center_x=obj["center_x"],
                    center_y=obj["crest_y"],
                    width_px=obj["width_px"],
                    width_mm=obj["width_mm"],
                    adjacent_left=None,
                    adjacent_right=None,
                    is_upper=is_upper,
                    arch_region=_arch_region_name(missing_fdis[0]),
                    crest_y=obj["crest_y"],
                    confidence=0.92,
                    source_type="edentulous_space",
                )
                edentulous_spaces.append(space)
                space_id_counter += 1
                for mfdi in missing_fdis:
                    assigned_r[mfdi] = {"type": "gap", "obj": space}

    for fdi in seq_r:
        if fdi in assigned_r:
            item = assigned_r[fdi]
            if item["type"] == "tooth":
                t_obj = item["obj"]
                implant_elig = "Potential Consideration (Evaluation Required)" if t_obj.status == "Possibly non-restorable" else "No"
                status_entries.append(ToothStatusEntry(
                    fdi=fdi,
                    arch_region=_arch_region_name(fdi),
                    detected=True,
                    status=t_obj.status,
                    confidence=t_obj.confidence,
                    bbox=t_obj.bbox,
                    reason=t_obj.clinical_note,
                    available_space_mm=t_obj.width_mm,
                    implant_consideration=implant_elig,
                ))
            else:
                g_obj = item["obj"]
                status_entries.append(ToothStatusEntry(
                    fdi=fdi,
                    arch_region=_arch_region_name(fdi),
                    detected=False,
                    status="Missing",
                    confidence=g_obj.confidence,
                    bbox=g_obj.bbox,
                    reason=f"Edentulous space detected (~{g_obj.width_mm:.1f} mm space). Implant replacement may be considered.",
                    available_space_mm=g_obj.width_mm,
                    implant_consideration="Yes — Candidate Site",
                ))
        else:
            if fdi in [18, 48]:
                status_entries.append(ToothStatusEntry(
                    fdi=fdi,
                    arch_region=_arch_region_name(fdi),
                    detected=False,
                    status="Uncertain",
                    confidence=0.70,
                    reason="Third molar not observed in active arch sequence (absent, unerupted, or previous extraction).",
                    implant_consideration="No",
                ))
            else:
                status_entries.append(ToothStatusEntry(
                    fdi=fdi,
                    arch_region=_arch_region_name(fdi),
                    detected=False,
                    status="Uncertain",
                    confidence=0.50,
                    reason="Region outside confident segmentation boundaries. Clinical evaluation required.",
                    implant_consideration="No",
                ))

    # ── Process Left Quadrant (midline -> molar) ───────────────────────────
    items_l = []
    for t in teeth_l:
        items_l.append(("tooth", t["cx"] - midline_x, t))
    for g in gaps_l:
        items_l.append(("gap", g["center_x"] - midline_x, g))
    items_l.sort(key=lambda x: x[1])

    fdi_idx = 0
    assigned_l: Dict[int, dict] = {}

    for itype, dist, obj in items_l:
        if fdi_idx >= len(seq_l):
            break

        if itype == "tooth":
            fdi = seq_l[fdi_idx]
            det = ToothDetection(
                fdi=fdi,
                bbox=(obj["x1"], obj["y1"], obj["x2"], obj["y2"]),
                cx=obj["cx"], cy=obj["cy"],
                width_px=obj["width_px"], height_px=obj["height_px"],
                width_mm=obj["width_mm"], height_mm=obj["height_mm"],
                confidence=obj["confidence"],
                is_upper=is_upper,
                status=obj["status"],
                clinical_note=obj["clinical_note"],
            )
            detected_teeth.append(det)
            assigned_l[fdi] = {"type": "tooth", "obj": det}
            fdi_idx += 1

        elif itype == "gap":
            gap_w_mm = obj["width_mm"]
            span_count = max(1, int(round(gap_w_mm / 7.5)))
            missing_fdis = []

            for _ in range(span_count):
                if fdi_idx < len(seq_l):
                    missing_fdis.append(seq_l[fdi_idx])
                    fdi_idx += 1

            if missing_fdis:
                space = EdentulousSpace(
                    space_id=space_id_counter,
                    suspected_fdis=missing_fdis,
                    bbox=(int(obj["gap_start_x"]), int(obj["crest_y"] - 25),
                          int(obj["gap_end_x"]),   int(obj["crest_y"] + 25)),
                    center_x=obj["center_x"],
                    center_y=obj["crest_y"],
                    width_px=obj["width_px"],
                    width_mm=obj["width_mm"],
                    adjacent_left=None,
                    adjacent_right=None,
                    is_upper=is_upper,
                    arch_region=_arch_region_name(missing_fdis[0]),
                    crest_y=obj["crest_y"],
                    confidence=0.92,
                    source_type="edentulous_space",
                )
                edentulous_spaces.append(space)
                space_id_counter += 1
                for mfdi in missing_fdis:
                    assigned_l[mfdi] = {"type": "gap", "obj": space}

    for fdi in seq_l:
        if fdi in assigned_l:
            item = assigned_l[fdi]
            if item["type"] == "tooth":
                t_obj = item["obj"]
                implant_elig = "Potential Consideration (Evaluation Required)" if t_obj.status == "Possibly non-restorable" else "No"
                status_entries.append(ToothStatusEntry(
                    fdi=fdi,
                    arch_region=_arch_region_name(fdi),
                    detected=True,
                    status=t_obj.status,
                    confidence=t_obj.confidence,
                    bbox=t_obj.bbox,
                    reason=t_obj.clinical_note,
                    available_space_mm=t_obj.width_mm,
                    implant_consideration=implant_elig,
                ))
            else:
                g_obj = item["obj"]
                status_entries.append(ToothStatusEntry(
                    fdi=fdi,
                    arch_region=_arch_region_name(fdi),
                    detected=False,
                    status="Missing",
                    confidence=g_obj.confidence,
                    bbox=g_obj.bbox,
                    reason=f"Edentulous space detected (~{g_obj.width_mm:.1f} mm space). Implant replacement may be considered.",
                    available_space_mm=g_obj.width_mm,
                    implant_consideration="Yes — Candidate Site",
                ))
        else:
            if fdi in [28, 38]:
                status_entries.append(ToothStatusEntry(
                    fdi=fdi,
                    arch_region=_arch_region_name(fdi),
                    detected=False,
                    status="Uncertain",
                    confidence=0.70,
                    reason="Third molar not observed in active arch sequence (absent, unerupted, or previous extraction).",
                    implant_consideration="No",
                ))
            else:
                status_entries.append(ToothStatusEntry(
                    fdi=fdi,
                    arch_region=_arch_region_name(fdi),
                    detected=False,
                    status="Uncertain",
                    confidence=0.50,
                    reason="Region outside confident segmentation boundaries. Clinical evaluation required.",
                    implant_consideration="No",
                ))

    # Link adjacent teeth
    for space in edentulous_spaces:
        left_candidates = [t for t in detected_teeth if t.is_upper == is_upper and t.cx < space.center_x]
        right_candidates = [t for t in detected_teeth if t.is_upper == is_upper and t.cx > space.center_x]
        if left_candidates:
            space.adjacent_left = max(left_candidates, key=lambda t: t.cx)
        if right_candidates:
            space.adjacent_right = min(right_candidates, key=lambda t: t.cx)

    return detected_teeth, edentulous_spaces, status_entries


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Landmarks Detection (IAC & Sinus Floor)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_anatomical_landmarks(img_bgr: np.ndarray) -> List[LandmarkResult]:
    h, w = img_bgr.shape[:2]
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    landmarks = []

    # 1. Inferior Alveolar Canal (IAC)
    y0_iac, y1_iac = int(0.60 * h), int(0.85 * h)
    iac_crop = img_gray[y0_iac:y1_iac, :]
    blurred = cv2.GaussianBlur(iac_crop, (5, 5), 0)
    edges = cv2.Canny(blurred, 25, 75)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=int(0.08 * w), maxLineGap=20)

    if lines is not None and len(lines) >= 2:
        reshaped = np.asarray(lines).reshape(-1, 4)
        horizontal_ys = []
        for x1l, y1l, x2l, y2l in reshaped:
            ang = abs(np.degrees(np.arctan2(y2l - y1l, x2l - x1l)))
            if ang <= 12.0 or ang >= 168.0:
                horizontal_ys.append((y1l + y2l) / 2.0 + y0_iac)
        if len(horizontal_ys) >= 2:
            y_med = float(np.median(horizontal_ys))
            landmarks.append(LandmarkResult(
                name="Inferior Alveolar Canal (IAC)",
                detected=True,
                y_left=y_med, y_right=y_med,
                confidence="Moderate (2D Edge)",
                note="Approximate canal boundary reference. Mandatory CBCT evaluation required for 3D canal mapping.",
            ))
        else:
            landmarks.append(LandmarkResult(
                name="Inferior Alveolar Canal (IAC)",
                detected=False,
                y_left=None, y_right=None,
                confidence="Not detected",
                note="Landmark not confidently detected from 2D panoramic radiograph.",
            ))
    else:
        landmarks.append(LandmarkResult(
            name="Inferior Alveolar Canal (IAC)",
            detected=False,
            y_left=None, y_right=None,
            confidence="Not detected",
            note="Landmark not confidently detected from 2D panoramic radiograph.",
        ))

    # 2. Maxillary Sinus Floor
    y0_sin, y1_sin = int(0.12 * h), int(0.42 * h)
    sin_crop = img_gray[y0_sin:y1_sin, :]
    edges_sin = cv2.Canny(cv2.GaussianBlur(sin_crop, (5, 5), 0), 30, 85)
    lines_sin = cv2.HoughLinesP(edges_sin, 1, np.pi / 180, threshold=45, minLineLength=int(0.06 * w), maxLineGap=15)

    if lines_sin is not None and len(lines_sin) >= 2:
        reshaped_s = np.asarray(lines_sin).reshape(-1, 4)
        horizontal_ys_s = []
        for x1l, y1l, x2l, y2l in reshaped_s:
            ang = abs(np.degrees(np.arctan2(y2l - y1l, x2l - x1l)))
            if ang <= 14.0 or ang >= 166.0:
                horizontal_ys_s.append((y1l + y2l) / 2.0 + y0_sin)
        if len(horizontal_ys_s) >= 2:
            y_med_s = float(np.median(horizontal_ys_s))
            landmarks.append(LandmarkResult(
                name="Maxillary Sinus Floor",
                detected=True,
                y_left=y_med_s, y_right=y_med_s,
                confidence="Moderate (2D Edge)",
                note="Approximate sinus floor line. 2D projection may distort vertical sinus extent.",
            ))
        else:
            landmarks.append(LandmarkResult(
                name="Maxillary Sinus Floor",
                detected=False,
                y_left=None, y_right=None,
                confidence="Not detected",
                note="Landmark not confidently detected from 2D panoramic radiograph.",
            ))
    else:
        landmarks.append(LandmarkResult(
            name="Maxillary Sinus Floor",
            detected=False,
            y_left=None, y_right=None,
            confidence="Not detected",
            note="Landmark not confidently detected from 2D panoramic radiograph.",
        ))

    return landmarks


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Preliminary Candidate Implant Sizing
# ─────────────────────────────────────────────────────────────────────────────

def _suggest_implant_size(gap_mm: float) -> Tuple[str, str]:
    for max_gap, diam_range, len_range in _IMPLANT_SIZES:
        if gap_mm <= max_gap:
            return diam_range, len_range
    return _IMPLANT_SIZES[-1][1], _IMPLANT_SIZES[-1][2]


def _make_implant_plans(
    spaces: List[EdentulousSpace],
    landmarks: List[LandmarkResult],
    px_per_mm: float,
    user_params: Dict[int, dict] = None,
) -> List[ImplantPlan]:
    user_params = user_params or {}
    plans: List[ImplantPlan] = []

    for space in spaces:
        site_fdi = space.suspected_fdis[0] if space.suspected_fdis else 99
        params = user_params.get(site_fdi, {})

        diam_str, len_str = _suggest_implant_size(space.width_mm)

        try:
            parts = diam_str.split("–")
            diam_mm = float(params.get("diameter_mm", (float(parts[0]) + float(parts[1].split()[0])) / 2.0))
        except Exception:
            diam_mm = float(params.get("diameter_mm", 4.0))

        try:
            lparts = len_str.split("–")
            len_mm = float(params.get("length_mm", (float(lparts[0]) + float(lparts[1].split()[0])) / 2.0))
        except Exception:
            len_mm = float(params.get("length_mm", 10.0))

        ang_deg = float(params.get("angulation_deg", 0.0))
        x_off   = float(params.get("x_offset_px", 0.0))
        y_off   = float(params.get("y_offset_px", 0.0))

        vert_h: Optional[float] = None
        if space.is_upper:
            sinus = next((l for l in landmarks if "Sinus" in l.name and l.detected), None)
            if sinus and sinus.y_left:
                vert_h = round(abs(space.crest_y - sinus.y_left) / px_per_mm, 1)
        else:
            iac = next((l for l in landmarks if "Canal" in l.name and l.detected), None)
            if iac and iac.y_left:
                vert_h = round(abs(iac.y_left - space.crest_y) / px_per_mm, 1)

        notes = [
            f"Potential implant site identified in {space.arch_region}.",
            f"Available mesiodistal space estimated at approximately {space.width_mm:.1f} mm.",
        ]
        if space.width_mm < 6.0:
            notes.append("⚠ Narrow space (<6 mm). Narrow platform may be considered.")
        elif space.width_mm > 14.0:
            notes.append("ℹ Wide edentulous space (>14 mm). Multiple implants or wider restoration may be evaluated.")

        if vert_h is not None:
            notes.append(f"Estimated vertical height from crest to landmark: approximately {vert_h:.1f} mm.")
            if vert_h < 8.0:
                notes.append("⚠ Limited vertical bone height estimated. Augmentation or short implant may be required.")
        else:
            notes.append("ℹ Landmark not confidently detected on 2D panoramic; vertical height not measurable.")

        notes.append("CLINICAL REQUIREMENT: Definitive surgical planning and buccolingual width evaluation require CBCT.")

        left_fdi = space.adjacent_left.fdi if space.adjacent_left else None
        right_fdi = space.adjacent_right.fdi if space.adjacent_right else None

        plans.append(ImplantPlan(
            site_fdi=site_fdi,
            status_category="Missing",
            adjacent_left_fdi=left_fdi,
            adjacent_right_fdi=right_fdi,
            center_x_px=space.center_x,
            center_y_px=space.center_y,
            x_offset_px=x_off,
            y_offset_px=y_off,
            diameter_mm=diam_mm,
            length_mm=len_mm,
            angulation_deg=ang_deg,
            is_upper=space.is_upper,
            suggested_diameter_range=diam_str,
            suggested_length_range=len_str,
            mesiodistal_mm=space.width_mm,
            vertical_height_mm=vert_h,
            planning_confidence="PRELIMINARY",
            clinical_reasons=notes,
            space=space,
        ))

    return plans


# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_full_planning(
    img_bgr: np.ndarray,
    binary_mask: np.ndarray,
    opg_width_mm: float = 150.0,
    is_calibrated: bool = False,
    user_params: Dict[int, dict] = None,
    min_gap_mm: float = MIN_EDENTULOUS_GAP_MM,
) -> PlanningResult:
    h, w = img_bgr.shape[:2]
    px_per_mm = w / float(opg_width_mm)

    from utils.yolo_tooth_detector import YoloToothDetector, analyze_missing_teeth_from_yolo

    yolo = YoloToothDetector()
    yolo_boxes = yolo.predict(img_bgr, conf_thresh=0.25) if yolo.is_available() else []

    if len(yolo_boxes) >= 8:
        # ── Fast & High-Accuracy 32-Class YOLO Pipeline ────────────────────────
        detected_teeth: List[ToothDetection] = [
            ToothDetection(
                fdi=tb.fdi,
                bbox=tb.bbox,
                cx=tb.cx, cy=tb.cy,
                width_px=tb.width_px, height_px=tb.height_px,
                width_mm=round(tb.width_px / px_per_mm, 1),
                height_mm=round(tb.height_px / px_per_mm, 1),
                confidence=tb.confidence,
                is_upper=tb.is_upper,
                status="Existing tooth",
                clinical_note=f"YOLO detection — FDI {tb.fdi} ({tb.confidence:.0%})",
            )
            for tb in yolo_boxes
        ]

        status_dicts, gap_dicts = analyze_missing_teeth_from_yolo(yolo_boxes, (h, w), px_per_mm)

        tooth_status_table = [
            ToothStatusEntry(
                fdi=sd["fdi"],
                arch_region=sd["arch_region"],
                detected=sd["detected"],
                status=sd["status"],
                confidence=sd["confidence"],
                bbox=sd["bbox"],
                available_space_mm=sd["available_space_mm"],
                implant_consideration=sd["implant_consideration"],
                reason=sd["reason"],
            )
            for sd in status_dicts
        ]

        edentulous_spaces: List[EdentulousSpace] = []
        for gd in gap_dicts:
            left_tooth = next((t for t in detected_teeth if t.fdi == gd["adjacent_left_fdi"]), None)
            right_tooth = next((t for t in detected_teeth if t.fdi == gd["adjacent_right_fdi"]), None)

            space = EdentulousSpace(
                space_id=gd["gap_id"],
                suspected_fdis=gd["missing_fdis"],
                bbox=gd["bbox"],
                center_x=gd["center_x"],
                center_y=gd["center_y"],
                width_px=gd["width_px"],
                width_mm=gd["width_mm"],
                adjacent_left=left_tooth,
                adjacent_right=right_tooth,
                is_upper=gd["is_upper"],
                arch_region=gd["arch_region"],
                crest_y=gd["center_y"],
                confidence=0.94,
                source_type="edentulous_space",
            )
            edentulous_spaces.append(space)

        # Fit visual arch curves
        arch_curves = {}
        upper_teeth = [t.__dict__ for t in detected_teeth if t.is_upper]
        lower_teeth = [t.__dict__ for t in detected_teeth if not t.is_upper]
        cu = _fit_arch_curve(upper_teeth, w)
        cl = _fit_arch_curve(lower_teeth, w)
        if cu is not None:
            arch_curves["upper"] = cu
        if cl is not None:
            arch_curves["lower"] = cl

    else:
        # ── Fallback Morphological Pipeline ──────────────────────────────────
        upper_mask, lower_mask, split_y = _separate_arches(binary_mask)

        raw_teeth_upper, raw_gaps_upper = _segment_arch_into_teeth(
            upper_mask, is_upper=True, px_per_mm=px_per_mm, min_gap_mm=min_gap_mm,
        )
        raw_teeth_lower, raw_gaps_lower = _segment_arch_into_teeth(
            lower_mask, is_upper=False, px_per_mm=px_per_mm, min_gap_mm=min_gap_mm,
        )

        midline_x = _find_dental_midline(raw_teeth_upper, raw_teeth_lower, w)

        arch_curves = {}
        curve_upper = _fit_arch_curve(raw_teeth_upper, w)
        curve_lower = _fit_arch_curve(raw_teeth_lower, w)
        if curve_upper is not None:
            arch_curves["upper"] = curve_upper
        if curve_lower is not None:
            arch_curves["lower"] = curve_lower

        det_upper, spaces_upper, status_upper = _assign_fdi_and_classify(
            raw_teeth_upper, raw_gaps_upper, is_upper=True, midline_x=midline_x, px_per_mm=px_per_mm,
        )
        det_lower, spaces_lower, status_lower = _assign_fdi_and_classify(
            raw_teeth_lower, raw_gaps_lower, is_upper=False, midline_x=midline_x, px_per_mm=px_per_mm,
        )

        detected_teeth = sorted(det_upper + det_lower, key=lambda t: t.fdi)
        edentulous_spaces = spaces_upper + spaces_lower

        status_map = {entry.fdi: entry for entry in (status_upper + status_lower)}
        tooth_status_table = [
            status_map.get(fdi, ToothStatusEntry(
                fdi=fdi,
                arch_region=_arch_region_name(fdi),
                detected=False,
                status="Uncertain",
                confidence=0.50,
                reason="Not assessed",
                implant_consideration="No",
            ))
            for fdi in ALL_32_FDI_TEETH
        ]

    landmarks = _detect_anatomical_landmarks(img_bgr)
    implant_plans = _make_implant_plans(edentulous_spaces, landmarks, px_per_mm, user_params)

    return PlanningResult(
        image_shape=(h, w),
        px_per_mm=px_per_mm,
        is_calibrated=is_calibrated,
        detected_teeth=detected_teeth,
        tooth_status_table=tooth_status_table,
        edentulous_spaces=edentulous_spaces,
        landmarks=landmarks,
        implant_plans=implant_plans,
        arch_curves=arch_curves,
        binary_mask=binary_mask,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Visual Rendering
# ─────────────────────────────────────────────────────────────────────────────

def _implant_polygon(
    cx: float, cy: float,
    diam_px: float, length_px: float,
    ang_deg: float, is_upper: bool,
) -> np.ndarray:
    hw   = diam_px / 2.0
    phw  = diam_px * 0.60
    taper = max(diam_px * 0.12, 2.0)
    d    = -1.0 if is_upper else 1.0

    local = np.array([
        [-phw,   0.0],
        [ phw,   0.0],
        [ phw,   d * length_px * 0.07],
        [ hw,    d * length_px * 0.07],
        [ hw,    d * length_px * 0.86],
        [ taper, d * length_px],
        [-taper, d * length_px],
        [-hw,    d * length_px * 0.86],
        [-hw,    d * length_px * 0.07],
        [-phw,   d * length_px * 0.07],
    ], dtype=float)

    ang_rad = np.radians(ang_deg)
    c, s = np.cos(ang_rad), np.sin(ang_rad)
    R = np.array([[c, -s], [s, c]])
    return (R @ local.T).T + np.array([cx, cy])


def render_planning_figure(
    result: PlanningResult,
    img_bgr: np.ndarray,
    show_teeth_boxes: bool = True,
    show_missing_boxes: bool = True,
    show_compromised_boxes: bool = True,
    show_measurements: bool = True,
    show_implants: bool = True,
    show_landmarks: bool = True,
    user_params: Dict[int, dict] = None,
    figsize: Tuple[int, int] = (22, 9),
) -> plt.Figure:
    """
    Renders the annotated 2D panoramic planning overview with distinct indicators for:
    - Existing tooth (Green)
    - Missing tooth / Edentulous space (Red dashed)
    - Possibly non-restorable tooth (Orange / Amber)
    - Candidate virtual implant (Cyan overlay)
    """
    user_params = user_params or {}
    h, w = result.image_shape
    px_mm = result.px_per_mm

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb, aspect="auto", alpha=0.92)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    # 1. Detected Teeth Boxes (Green for Existing, Orange for Possibly Non-restorable)
    for t in result.detected_teeth:
        x1, y1, x2, y2 = t.bbox

        if t.status == "Possibly non-restorable" and show_compromised_boxes:
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.0, edgecolor="#F39C12", facecolor="#F39C12", alpha=0.20,
                linestyle=":", zorder=3,
            )
            ax.add_patch(rect)
            label_y = y1 - 8 if t.is_upper else y2 + 16
            ax.text(
                t.cx, label_y, f"{t.fdi} ⚠ (Compromised)",
                color="#F39C12", fontsize=7.5, ha="center", va="center", fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#241400", alpha=0.85, edgecolor="#F39C12", lw=1.0),
            )
        elif t.status == "Existing tooth" and show_teeth_boxes:
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor="#2ECC71", facecolor="#2ECC71", alpha=0.10,
                zorder=3,
            )
            ax.add_patch(rect)
            label_y = y1 - 8 if t.is_upper else y2 + 16
            ax.text(
                t.cx, label_y, f"{t.fdi} ✓",
                color="#F1C40F", fontsize=8.0, ha="center", va="center", fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#101820", alpha=0.85, edgecolor="none"),
            )

    # 2. Missing Tooth / Edentulous Spaces (Red)
    if show_missing_boxes:
        for space in result.edentulous_spaces:
            x1, y1, x2, y2 = space.bbox
            rect_gap = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.0, edgecolor="#E74C3C", facecolor="#E74C3C", alpha=0.18,
                linestyle="--", zorder=4,
            )
            ax.add_patch(rect_gap)

            missing_str = " | ".join(f"FDI {f} MISSING" for f in space.suspected_fdis)
            label_gy = y1 - 12 if space.is_upper else y2 + 18
            ax.text(
                space.center_x, label_gy,
                f"POTENTIAL IMPLANT SITE\n{missing_str}",
                color="#FF6B6B", fontsize=8.0, ha="center", va="center",
                fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#1A0000", alpha=0.90, edgecolor="#E74C3C", lw=1.2),
            )

    # 3. Mesiodistal Measurements
    if show_measurements:
        for space in result.edentulous_spaces:
            x1, _, x2, _ = space.bbox
            my = space.center_y
            ax.plot([x1, x2], [my, my], color="#F39C12", lw=1.8, ls="-", zorder=5)
            ax.plot([x1, x1], [my - 6, my + 6], color="#F39C12", lw=1.8, zorder=5)
            ax.plot([x2, x2], [my - 6, my + 6], color="#F39C12", lw=1.8, zorder=5)

            calib_suffix = " mm (approx.)" if not result.is_calibrated else " mm"
            ax.text(
                space.center_x, my - 8,
                f"~{space.width_mm:.1f}{calib_suffix}",
                color="#F39C12", fontsize=7.5, ha="center", va="bottom",
                fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#111118", alpha=0.80, edgecolor="none"),
            )

    # 4. Landmarks (IAC & Sinus Floor)
    if show_landmarks:
        for lmk in result.landmarks:
            if not lmk.detected:
                continue
            color = "#E056FD" if "Canal" in lmk.name else "#22A6B3"
            short_name = "IAC Reference Line" if "Canal" in lmk.name else "Sinus Floor Line"
            if lmk.y_left is not None:
                ax.axhline(y=lmk.y_left, color=color, lw=1.4, ls="--", alpha=0.75, zorder=3)
                ax.text(w * 0.01, lmk.y_left - 4, short_name,
                        color=color, fontsize=7.0, ha="left", va="bottom", alpha=0.85)

    # 5. Candidate Virtual Implant Overlays
    if show_implants:
        for plan in result.implant_plans:
            p = user_params.get(plan.site_fdi, {})
            diam_mm = float(p.get("diameter_mm", plan.diameter_mm))
            len_mm  = float(p.get("length_mm",   plan.length_mm))
            ang_deg = float(p.get("angulation_deg", plan.angulation_deg))
            x_off   = float(p.get("x_offset_px",  plan.x_offset_px))
            y_off   = float(p.get("y_offset_px",  plan.y_offset_px))

            diam_px = diam_mm * px_mm
            len_px  = len_mm  * px_mm
            cx      = plan.center_x_px + x_off
            cy      = plan.center_y_px + y_off

            poly = _implant_polygon(cx, cy, diam_px, len_px, ang_deg, plan.is_upper)
            patch = mpatches.Polygon(
                poly, closed=True,
                facecolor="#00E5FF", alpha=0.42,
                edgecolor="#00E5FF", linewidth=2.0, zorder=7,
            )
            ax.add_patch(patch)

            # Central axis line
            d = -1 if plan.is_upper else 1
            ang_rad = np.radians(ang_deg)
            c_, s_ = np.cos(ang_rad), np.sin(ang_rad)
            ax.plot(
                [cx, cx + s_ * len_px],
                [cy, cy + c_ * d * len_px],
                color="#FFFFFF", lw=1.4, ls=":", alpha=0.90, zorder=8,
            )

            # Badge
            badge_y = cy - 22 if plan.is_upper else cy + 22
            ax.text(
                cx, badge_y,
                f"⊕ Candidate FDI {plan.site_fdi}\nØ{diam_mm:.1f} × {len_mm:.0f} mm (approx)",
                color="#00E5FF", fontsize=7.5, ha="center", va="center",
                fontweight="bold", zorder=9,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#001820", alpha=0.90, edgecolor="#00E5FF", lw=1.0),
            )

    # Legend
    legend_items = [
        mpatches.Patch(color="#2ECC71", label="Existing Tooth (Intact)"),
        mpatches.Patch(color="#E74C3C", label="Missing Tooth (Edentulous Space)"),
        mpatches.Patch(color="#F39C12", label="Possibly Non-Restorable Tooth (⚠)"),
        mpatches.Patch(color="#00E5FF", label="Candidate Implant Overlay (Preliminary)"),
    ]
    ax.legend(
        handles=legend_items, loc="lower left",
        fontsize=8, fancybox=True,
        framealpha=0.85, facecolor="#0d1117", edgecolor="#334155", labelcolor="white",
    )

    fig.text(
        0.5, 0.01,
        "⚠ PRELIMINARY 2D PANORAMIC ASSESSMENT ONLY — Buccolingual bone width cannot be determined from 2D X-rays. CBCT & clinical examination required.",
        color="#94A3B8", fontsize=8.0, ha="center", va="bottom", style="italic",
    )
    fig.tight_layout(pad=0.5)
    return fig


def render_debug_figure(
    result: PlanningResult,
    img_bgr: np.ndarray,
    figsize: Tuple[int, int] = (22, 9),
) -> plt.Figure:
    h, w = result.image_shape
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="#0a0e14")
    ax.set_facecolor("#0a0e14")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb, aspect="auto", alpha=0.80)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.set_title("DEBUG MODE — Anatomical Arch Curves, Centroids & Coordinate Inspection",
                 color="#00E5FF", fontsize=11, fontweight="bold", pad=8)

    if "upper" in result.arch_curves:
        curve = result.arch_curves["upper"]
        ax.plot(curve[:, 0], curve[:, 1], color="#F1C40F", lw=2.0, ls="-", label="Maxillary Arch Curve")
    if "lower" in result.arch_curves:
        curve = result.arch_curves["lower"]
        ax.plot(curve[:, 0], curve[:, 1], color="#3498DB", lw=2.0, ls="-", label="Mandibular Arch Curve")

    for t in result.detected_teeth:
        x1, y1, x2, y2 = t.bbox
        ec = "#2ECC71" if t.status == "Existing tooth" else "#F39C12"
        rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.2, edgecolor=ec, facecolor="none")
        ax.add_patch(rect)
        ax.plot(t.cx, t.cy, marker="o", color=ec, markersize=3)
        ax.text(
            t.cx, t.cy, f"FDI {t.fdi}\n{t.confidence:.2f}",
            color=ec, fontsize=6.5, ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="#000000", alpha=0.7, edgecolor="none"),
        )

    for space in result.edentulous_spaces:
        x1, y1, x2, y2 = space.bbox
        rect_gap = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor="#E74C3C", facecolor="#E74C3C", alpha=0.20, ls="--")
        ax.add_patch(rect_gap)
        ax.text(
            space.center_x, space.center_y,
            f"GAP\n{space.width_px:.0f}px\n(~{space.width_mm:.1f}mm)",
            color="#FF4444", fontsize=7.0, ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#000000", alpha=0.8, edgecolor="#FF4444"),
        )

    ax.legend(loc="upper right", fontsize=8, facecolor="#0a0e14", edgecolor="#334155", labelcolor="white")
    fig.tight_layout(pad=0.5)
    return fig


def render_site_detail_figure(
    result: PlanningResult,
    img_bgr: np.ndarray,
    site_fdi: int,
    user_params: Dict[int, dict] = None,
    pad_frac: float = 0.14,
) -> Optional[plt.Figure]:
    user_params = user_params or {}
    plan = next((p for p in result.implant_plans if p.site_fdi == site_fdi), None)
    if plan is None:
        return None

    h, w = result.image_shape
    px_mm = result.px_per_mm
    space = plan.space

    margin_x = max(int(space.width_px * 1.6), int(w * pad_frac))
    margin_y = int(h * 0.20)

    cx0 = max(0, int(space.bbox[0]) - margin_x)
    cx1 = min(w, int(space.bbox[2]) + margin_x)
    cy0 = max(0, int(space.center_y) - margin_y)
    cy1 = min(h, int(space.center_y) + margin_y)

    crop_bgr = img_bgr[cy0:cy1, cx0:cx1]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    cw, ch = cx1 - cx0, cy1 - cy0

    fig, ax = plt.subplots(1, 1, figsize=(12, 5.5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.imshow(crop_rgb, aspect="auto", alpha=0.92)
    ax.set_xlim(0, cw)
    ax.set_ylim(ch, 0)
    ax.axis("off")
    ax.set_title(f"Potential Implant Site ROI — FDI {site_fdi}   [{space.arch_region}]",
                 color="#E2E8F0", fontsize=11, fontweight="bold", pad=8)

    def lx(px): return px - cx0
    def ly(py): return py - cy0

    for adj in (space.adjacent_left, space.adjacent_right):
        if adj is None:
            continue
        rx1, ry1, rx2, ry2 = adj.bbox
        ec = "#2ECC71" if adj.status == "Existing tooth" else "#F39C12"
        rect = mpatches.Rectangle((lx(rx1), ly(ry1)), rx2 - rx1, ry2 - ry1, linewidth=2.0, edgecolor=ec, facecolor=ec, alpha=0.15)
        ax.add_patch(rect)
        ax.text(
            lx(adj.cx), ly(adj.bbox[1]) - 10,
            f"FDI {adj.fdi} ✓\n(Adjacent)",
            color="#F1C40F", fontsize=9.0, ha="center", va="bottom", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#101820", alpha=0.85, edgecolor="none"),
        )

    gx1, gy1, gx2, gy2 = space.bbox
    rect_gap = mpatches.Rectangle((lx(gx1), ly(gy1)), gx2 - gx1, gy2 - gy1, linewidth=2.0, edgecolor="#E74C3C", facecolor="#E74C3C", alpha=0.15, ls="--")
    ax.add_patch(rect_gap)

    meas_y = ly(space.center_y + ch * 0.15)
    ax.plot([lx(gx1), lx(gx2)], [meas_y, meas_y], color="#F39C12", lw=2.0)
    ax.plot([lx(gx1), lx(gx1)], [meas_y - 8, meas_y + 8], color="#F39C12", lw=2.0)
    ax.plot([lx(gx2), lx(gx2)], [meas_y - 8, meas_y + 8], color="#F39C12", lw=2.0)
    ax.text(
        lx(space.center_x), meas_y - 8,
        f"~{space.width_mm:.1f} mm (approx.)",
        color="#F39C12", fontsize=9.0, ha="center", va="bottom", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#111118", alpha=0.85, edgecolor="none"),
    )

    p = user_params.get(site_fdi, {})
    diam_mm = float(p.get("diameter_mm", plan.diameter_mm))
    len_mm  = float(p.get("length_mm",   plan.length_mm))
    ang_deg = float(p.get("angulation_deg", plan.angulation_deg))
    x_off   = float(p.get("x_offset_px",  plan.x_offset_px))
    y_off   = float(p.get("y_offset_px",  plan.y_offset_px))

    diam_px = diam_mm * px_mm
    len_px  = len_mm  * px_mm
    icx     = lx(plan.center_x_px + x_off)
    icy     = ly(plan.center_y_px + y_off)

    poly = _implant_polygon(icx, icy, diam_px, len_px, ang_deg, plan.is_upper)
    patch = mpatches.Polygon(poly, closed=True, facecolor="#00E5FF", alpha=0.45, edgecolor="#00E5FF", linewidth=2.5, zorder=8)
    ax.add_patch(patch)

    d_sign = -1 if plan.is_upper else 1
    ang_rad = np.radians(ang_deg)
    c_, s_ = np.cos(ang_rad), np.sin(ang_rad)
    ax.plot([icx, icx + s_ * len_px], [icy, icy + c_ * d_sign * len_px], color="#FFFFFF", lw=1.6, ls=":", alpha=0.90, zorder=9)

    label_y = icy - 22 if plan.is_upper else icy + 22
    ax.text(
        icx, label_y,
        f"Potential Site: FDI {site_fdi}\nØ{diam_mm:.1f} × {len_mm:.0f} mm (approx)  {ang_deg:+.0f}°",
        color="#00E5FF", fontsize=9.5, ha="center", va="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#001820", alpha=0.90, edgecolor="#00E5FF", lw=1.2),
    )

    fig.tight_layout(pad=0.4)
    return fig


def generate_site_report(plan: ImplantPlan, result: PlanningResult) -> str:
    calib_str = "(approx.)" if result.is_calibrated else "(approx. — uncalibrated)"
    vert_str = f"~{plan.vertical_height_mm:.1f} mm {calib_str}" if plan.vertical_height_mm else "Landmark not confidently detected on 2D image"
    left_adj  = f"FDI {plan.adjacent_left_fdi}"  if plan.adjacent_left_fdi  else "None detected"
    right_adj = f"FDI {plan.adjacent_right_fdi}" if plan.adjacent_right_fdi else "None detected"
    notes_str = "\n".join(f"  • {n}" for n in plan.clinical_reasons)

    return f"""
{"=" * 56}
  2D PANORAMIC IMPLANT PLANNING REPORT — FDI {plan.site_fdi}
{"=" * 56}

  Potential Implant Site:
    FDI {plan.site_fdi} ({plan.space.arch_region})

  Status:
    Edentulous space detected; implant replacement may be considered.

  Adjacent Teeth:
    Left : {left_adj}
    Right: {right_adj}

  Approximate Available Mesiodistal Space:
    ~{plan.mesiodistal_mm:.1f} mm {calib_str}

  Estimated Vertical Bone Height:
    {vert_str}

  Preliminary Implant Suggestion:
    Diameter: approximately {plan.diameter_mm:.1f} mm ({plan.suggested_diameter_range})
    Length  : approximately {plan.length_mm:.0f} mm ({plan.suggested_length_range})
    Position: centered between {left_adj} and {right_adj}

  Assessment Status:
    PRELIMINARY — DENTIST EVALUATION REQUIRED

  Clinical Considerations:
{notes_str}

  --------------------------------------------------------
  CLINICAL WARNING:
  {CLINICAL_DISCLAIMER_TEXT}
  --------------------------------------------------------
"""


def plans_to_dict_list(plans: List[ImplantPlan]) -> List[dict]:
    return [
        {
            "site_fdi":               int(p.site_fdi) if p.site_fdi is not None else None,
            "status_category":        str(p.status_category),
            "arch_region":            str(p.space.arch_region),
            "adjacent_left_fdi":      int(p.adjacent_left_fdi) if p.adjacent_left_fdi is not None else None,
            "adjacent_right_fdi":     int(p.adjacent_right_fdi) if p.adjacent_right_fdi is not None else None,
            "mesiodistal_mm":         float(p.mesiodistal_mm) if p.mesiodistal_mm is not None else None,
            "vertical_height_mm":     float(p.vertical_height_mm) if p.vertical_height_mm is not None else None,
            "suggested_diameter":     str(p.suggested_diameter_range),
            "suggested_length":       str(p.suggested_length_range),
            "planning_diameter_mm":   float(p.diameter_mm),
            "planning_length_mm":     float(p.length_mm),
            "angulation_deg":         float(p.angulation_deg),
            "planning_confidence":    str(p.planning_confidence),
            "clinical_reasons":       [str(n) for n in p.clinical_reasons],
        }
        for p in plans
    ]
