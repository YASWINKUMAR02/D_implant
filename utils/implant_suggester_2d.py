"""
implant_suggester_2d.py
=======================
AI Dental Implant Suggestion Engine for 2D Panoramic X-rays (OPG).

Given a binary teeth segmentation mask (from U-Net++), this module:
  1. Extracts individual tooth blobs via connected-component analysis.
  2. Fits a parametric dental arch curve through blob centroids.
  3. Detects edentulous gaps along the arch (missing tooth positions).
  4. Maps gap width → recommended implant diameter / type.
  5. Generates a richly annotated BGR overlay image for display.
  6. Returns structured suggestion data for the Streamlit UI table.

Typical clinical OPG pixel spacing assumption: ~150 mm field width.
The user may override this via the `image_width_mm` parameter.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Clinical Implant Size Lookup Table
# Maps edentulous gap width (mm) → (implant_diameter_mm, implant_length_mm, type_label)
# Based on standard implant selection guidelines.
# ─────────────────────────────────────────────────────────────────────────────
_IMPLANT_LOOKUP: List[Tuple[float, float, float, str]] = [
    # (max_gap_mm, diameter_mm, length_mm, type_label)
    (5.5,  3.3,  10.0, "Narrow Platform (NP)"),
    (6.5,  3.5,  10.0, "Narrow Platform (NP)"),
    (7.5,  4.0,  10.0, "Regular Platform (RP)"),
    (8.5,  4.0,  11.5, "Regular Platform (RP)"),
    (9.5,  4.5,  11.5, "Regular Platform (RP)"),
    (11.0, 5.0,  11.5, "Wide Platform (WP)"),
    (14.0, 5.0,  13.0, "Wide Platform (WP) — consider 2 implants"),
    (999., 5.0,  13.0, "Wide Platform (WP) — 2 implants recommended"),
]

# Minimum gap width to consider (< this is likely an inter-proximal space, not edentulous)
MIN_GAP_MM = 4.5
# Typical average adult mesiodistal tooth diameter (roughly 7 mm molars, 5.5 mm premolars)
# We use 6 mm as a practical separation threshold in normalised pixel space
_TOOTH_SEP_FRACTION = 0.045   # gap must be > 4.5% of image width to count


@dataclass
class ImplantSuggestion2D:
    """Represents a single 2D implant site suggestion on a panoramic OPG."""
    site_id: int                      # 1-indexed suggestion number
    arch_region: str                  # "Upper Left", "Upper Right", "Lower Left", "Lower Right"
    centroid_x_px: float              # X pixel of suggested implant center
    centroid_y_px: float              # Y pixel of suggested implant center
    gap_width_px: float               # edentulous gap width in pixels
    gap_width_mm: float               # edentulous gap width in mm (estimated)
    left_tooth_x_px: Optional[float]  # X pixel of tooth to the left of gap
    right_tooth_x_px: Optional[float] # X pixel of tooth to the right of gap
    recommended_diameter_mm: float    # recommended implant diameter
    recommended_length_mm: float      # recommended implant length
    implant_type: str                 # descriptive type label
    confidence: float                 # 0–1 score (based on gap regularity)
    notes: str = ""


def _get_implant_for_gap(gap_mm: float) -> Tuple[float, float, str]:
    """Look up recommended implant diameter/length/type for a given gap width in mm."""
    for max_gap, diam, length, label in _IMPLANT_LOOKUP:
        if gap_mm <= max_gap:
            return diam, length, label
    diam, length, label = _IMPLANT_LOOKUP[-1][1], _IMPLANT_LOOKUP[-1][2], _IMPLANT_LOOKUP[-1][3]
    return diam, length, label


def extract_tooth_blobs(
    mask: np.ndarray,
    min_blob_area_frac: float = 0.0005,
) -> List[dict]:
    """
    Extract individual tooth regions as connected components from binary mask.

    Args:
        mask: uint8 binary mask, teeth = 255 (or >0).
        min_blob_area_frac: minimum blob area as fraction of total image area.

    Returns:
        List of dicts with keys: cx, cy, x1, y1, x2, y2, area, width, height.
        Sorted left-to-right by cx.
    """
    bin_mask = (mask > 0).astype(np.uint8)

    # Morphological closing to merge fragmented teeth pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

    h, w = mask.shape[:2]
    min_area = min_blob_area_frac * h * w

    blobs = []
    for i in range(1, num_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x1 = stats[i, cv2.CC_STAT_LEFT]
        y1 = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]
        blobs.append({
            "cx": cx, "cy": cy,
            "x1": x1, "y1": y1,
            "x2": x1 + bw, "y2": y1 + bh,
            "area": area,
            "width": bw, "height": bh,
        })

    blobs.sort(key=lambda b: b["cx"])
    return blobs


def split_upper_lower(blobs: List[dict], image_height: int) -> Tuple[List[dict], List[dict]]:
    """
    Split tooth blobs into upper and lower arch based on vertical position.
    In OPG images upper teeth occupy the top ~50% of the image.
    """
    mid_y = image_height * 0.50
    upper = [b for b in blobs if b["cy"] < mid_y]
    lower = [b for b in blobs if b["cy"] >= mid_y]
    return upper, lower


def detect_gaps_in_arch(
    blobs: List[dict],
    image_width: int,
    image_width_mm: float = 150.0,
    is_upper: bool = True,
) -> List[dict]:
    """
    Detect edentulous gaps between adjacent tooth blobs along a single arch.

    Args:
        blobs: sorted left-to-right tooth blobs.
        image_width: pixel width of original image.
        image_width_mm: assumed physical width of the OPG (mm).
        is_upper: True for maxillary arch, False for mandibular.

    Returns:
        List of gap dicts: gap_x, gap_y, gap_width_px, gap_width_mm, left_blob, right_blob.
    """
    px_per_mm = image_width / image_width_mm
    min_gap_px = MIN_GAP_MM * px_per_mm
    min_gap_px_frac = _TOOTH_SEP_FRACTION * image_width

    # Use the larger of the two thresholds
    effective_min_gap_px = max(min_gap_px, min_gap_px_frac)

    gaps = []
    for i in range(len(blobs) - 1):
        left  = blobs[i]
        right = blobs[i + 1]

        # Gap is between right edge of left blob and left edge of right blob
        gap_start_x = left["x2"]
        gap_end_x   = right["x1"]
        gap_width_px = gap_end_x - gap_start_x

        if gap_width_px < effective_min_gap_px:
            continue  # too small — inter-proximal or measurement noise

        # Interpolated Y position at midpoint of gap
        t = 0.5
        gap_x = gap_start_x + t * gap_width_px
        gap_y = left["cy"] * (1 - t) + right["cy"] * t

        gap_width_mm = gap_width_px / px_per_mm

        gaps.append({
            "gap_x":         gap_x,
            "gap_y":         gap_y,
            "gap_start_x":   gap_start_x,
            "gap_end_x":     gap_end_x,
            "gap_width_px":  gap_width_px,
            "gap_width_mm":  gap_width_mm,
            "left_blob":     left,
            "right_blob":    right,
            "is_upper":      is_upper,
        })

    return gaps


def _arch_region_label(gap_x: float, image_width: int, is_upper: bool) -> str:
    """Map a pixel X position to a clinical arch quadrant label."""
    mid = image_width / 2.0
    jaw = "Upper" if is_upper else "Lower"
    # In OPG: patient's RIGHT appears on viewer's LEFT (radiographic convention)
    side = "Right" if gap_x < mid else "Left"
    return f"{jaw} {side}"


def suggest_implants_2d(
    mask: np.ndarray,
    image_width_mm: float = 150.0,
) -> List[ImplantSuggestion2D]:
    """
    Main entry point: detect edentulous gaps in a 2D panoramic OPG binary mask
    and suggest implants for each gap.

    Args:
        mask: uint8 binary mask (teeth pixels = 255 or > 0). Shape (H, W).
        image_width_mm: assumed physical width of the OPG in mm.

    Returns:
        List of ImplantSuggestion2D objects, ordered by arch (upper first) then L→R.
    """
    h, w = mask.shape[:2]
    blobs = extract_tooth_blobs(mask)

    if len(blobs) < 2:
        logger.warning("Too few tooth blobs detected for gap analysis.")
        return []

    upper_blobs, lower_blobs = split_upper_lower(blobs, h)

    upper_gaps = detect_gaps_in_arch(upper_blobs, w, image_width_mm, is_upper=True)
    lower_gaps = detect_gaps_in_arch(lower_blobs, w, image_width_mm, is_upper=False)

    all_gaps = upper_gaps + lower_gaps

    suggestions: List[ImplantSuggestion2D] = []
    site_id = 1

    for gap in all_gaps:
        gap_mm = gap["gap_width_mm"]
        diam, length, label = _get_implant_for_gap(gap_mm)

        # Confidence: based on how well the gap aligns with an expected tooth width
        # Peak confidence at ~7 mm (single molar), degrades for very wide/narrow gaps
        expected_single_tooth_mm = 7.0
        deviation = abs(gap_mm - expected_single_tooth_mm) / expected_single_tooth_mm
        confidence = float(np.clip(1.0 - deviation * 0.5, 0.30, 1.0))

        notes = ""
        if gap_mm > 14.0:
            notes = "Gap is very wide — 2 implants may be required."
        elif gap_mm < MIN_GAP_MM + 1.0:
            notes = "Borderline gap width — clinical evaluation recommended."

        arch_region = _arch_region_label(gap["gap_x"], w, gap["is_upper"])

        suggestions.append(ImplantSuggestion2D(
            site_id=site_id,
            arch_region=arch_region,
            centroid_x_px=gap["gap_x"],
            centroid_y_px=gap["gap_y"],
            gap_width_px=gap["gap_width_px"],
            gap_width_mm=round(gap_mm, 1),
            left_tooth_x_px=gap["left_blob"]["cx"],
            right_tooth_x_px=gap["right_blob"]["cx"],
            recommended_diameter_mm=diam,
            recommended_length_mm=length,
            implant_type=label,
            confidence=round(confidence, 2),
            notes=notes,
        ))
        site_id += 1

    return suggestions


# ─────────────────────────────────────────────────────────────────────────────
# Annotated Overlay Rendering
# ─────────────────────────────────────────────────────────────────────────────

# Color palette (BGR)
_COL_ARCH_UPPER = (255, 200, 60)    # amber — upper arch curve
_COL_ARCH_LOWER = (60, 220, 255)    # cyan  — lower arch curve
_COL_GAP_BOX    = (80, 80, 255)     # red   — edentulous gap box
_COL_IMPLANT    = (0, 230, 120)     # green — implant marker
_COL_TOOTH_MASK = (0, 200, 60)      # green — teeth overlay
_COL_TEXT_BG    = (15, 20, 30)      # near-black text bg


def _draw_arch_curve(
    canvas: np.ndarray,
    blobs: List[dict],
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Fit a degree-2 polynomial through blob centroids and draw it."""
    if len(blobs) < 3:
        return
    xs = np.array([b["cx"] for b in blobs])
    ys = np.array([b["cy"] for b in blobs])
    try:
        coeffs = np.polyfit(xs, ys, 2)
        poly = np.poly1d(coeffs)
        x_range = np.linspace(xs.min(), xs.max(), 300).astype(np.float32)
        y_range = poly(x_range).astype(np.float32)
        pts = np.column_stack([x_range, y_range]).astype(np.int32)
        # Clip to image bounds
        h, w = canvas.shape[:2]
        valid = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        pts = pts[valid]
        for i in range(len(pts) - 1):
            cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]), color, thickness, cv2.LINE_AA)
    except Exception:
        pass


def _put_label(
    canvas: np.ndarray,
    text: str,
    cx: int,
    cy: int,
    font_scale: float = 0.55,
    thickness: int = 1,
    text_color=(255, 255, 255),
    bg_color=_COL_TEXT_BG,
) -> None:
    """Draw a text label with a filled background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 3
    x0 = cx - tw // 2 - pad
    y0 = cy - th - pad
    x1 = cx + tw // 2 + pad
    y1 = cy + baseline + pad
    cv2.rectangle(canvas, (x0, y0), (x1, y1), bg_color, -1)
    cv2.putText(canvas, text, (cx - tw // 2, cy), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_implant_suggestions(
    orig_bgr: np.ndarray,
    mask: np.ndarray,
    suggestions: List[ImplantSuggestion2D],
    show_arch_curve: bool = True,
    show_tooth_overlay: bool = True,
    opacity: float = 0.35,
) -> np.ndarray:
    """
    Render an annotated BGR image with:
      - Semi-transparent green teeth overlay
      - Upper/lower arch polynomial curves
      - Cyan circles + site ID labels at each suggested implant location
      - Red bounding boxes marking edentulous gaps

    Args:
        orig_bgr: original BGR image (H, W, 3).
        mask: binary teeth mask (H, W), values 0 or 255.
        suggestions: list of ImplantSuggestion2D from suggest_implants_2d().
        show_arch_curve: draw fitted arch curves.
        show_tooth_overlay: draw semi-transparent green teeth overlay.
        opacity: transparency of teeth overlay (0=invisible, 1=solid).

    Returns:
        Annotated BGR image.
    """
    canvas = orig_bgr.copy()
    h, w = canvas.shape[:2]

    # 1. Semi-transparent teeth overlay
    if show_tooth_overlay:
        green_layer = np.zeros_like(canvas)
        green_layer[mask > 0] = [0, 200, 60]
        cv2.addWeighted(canvas, 1.0, green_layer, opacity, 0, canvas)

    # 2. Arch curves
    if show_arch_curve and suggestions:
        blobs = extract_tooth_blobs(mask)
        upper_blobs, lower_blobs = split_upper_lower(blobs, h)
        _draw_arch_curve(canvas, upper_blobs, _COL_ARCH_UPPER, thickness=2)
        _draw_arch_curve(canvas, lower_blobs, _COL_ARCH_LOWER, thickness=2)

    # 3. Gap boxes + implant markers
    for s in suggestions:
        cx = int(round(s.centroid_x_px))
        cy = int(round(s.centroid_y_px))

        # Gap bounding box (width = gap width, height based on arch height)
        gw = int(round(s.gap_width_px))
        gap_box_h = int(h * 0.06)  # ~6% of image height
        gap_x0 = int(round(s.centroid_x_px - gw / 2))
        gap_y0 = int(round(s.centroid_y_px - gap_box_h // 2))
        gap_x1 = gap_x0 + gw
        gap_y1 = gap_y0 + gap_box_h

        # Draw gap box — semi-transparent red
        gap_overlay = canvas.copy()
        cv2.rectangle(gap_overlay, (gap_x0, gap_y0), (gap_x1, gap_y1), (80, 80, 220), -1)
        cv2.addWeighted(canvas, 0.75, gap_overlay, 0.25, 0, canvas)
        cv2.rectangle(canvas, (gap_x0, gap_y0), (gap_x1, gap_y1), (80, 80, 220), 2, cv2.LINE_AA)

        # Implant circle (cyan)
        radius = max(int(s.gap_width_px * 0.20), 8)
        cv2.circle(canvas, (cx, cy), radius, _COL_IMPLANT, 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 3, _COL_IMPLANT, -1)

        # Cross-hair inside circle
        cv2.line(canvas, (cx - radius, cy), (cx + radius, cy), _COL_IMPLANT, 1, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy - radius), (cx, cy + radius), _COL_IMPLANT, 1, cv2.LINE_AA)

        # Site ID label above the circle
        label_y = max(gap_y0 - 6, 14)
        _put_label(
            canvas,
            f"#{s.site_id}  Ø{s.recommended_diameter_mm:.1f}×{s.recommended_length_mm:.0f}mm",
            cx, label_y,
            font_scale=0.45, thickness=1,
            text_color=(220, 255, 100),
        )

    # 4. Legend strip at bottom
    legend_h = 28
    legend = np.zeros((legend_h, w, 3), dtype=np.uint8)
    legend[:] = (20, 25, 35)
    legend_items = [
        ((0, 200, 60),   "Teeth Mask"),
        (_COL_ARCH_UPPER, "Upper Arch Curve"),
        (_COL_ARCH_LOWER, "Lower Arch Curve"),
        ((80, 80, 220),  "Edentulous Gap"),
        (_COL_IMPLANT,   "Suggested Implant Site"),
    ]
    x_off = 8
    for color, text in legend_items:
        cv2.rectangle(legend, (x_off, 8), (x_off + 12, 20), color, -1)
        cv2.putText(legend, text, (x_off + 15, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (200, 210, 220), 1, cv2.LINE_AA)
        x_off += len(text) * 7 + 30

    canvas = np.vstack([canvas, legend])
    return canvas


def suggestions_to_dict_list(suggestions: List[ImplantSuggestion2D]) -> List[dict]:
    """Convert suggestion list to JSON-serializable list of dicts for export."""
    return [
        {
            "site_id":                  s.site_id,
            "arch_region":              s.arch_region,
            "gap_width_mm":             s.gap_width_mm,
            "recommended_diameter_mm":  s.recommended_diameter_mm,
            "recommended_length_mm":    s.recommended_length_mm,
            "implant_type":             s.implant_type,
            "confidence":               s.confidence,
            "notes":                    s.notes,
        }
        for s in suggestions
    ]
