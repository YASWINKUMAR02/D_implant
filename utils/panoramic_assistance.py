"""
panoramic_assistance.py
=======================
Professional 2D Dental Implant-Assistance Module for Panoramic Radiographs (OPG).
Strictly 2D (No CBCT, No 3D meshes, No volumetric claims).

WORKFLOW:
PANORAMIC X-RAY
      ↓
1. TOOTH SEGMENTATION (Individual Tooth Instances, Masks, BBoxes, Centroids)
      ↓
2. FDI TOOTH NUMBERING (11-18, 21-28, 31-38, 41-48 with Arch Geometry)
      ↓
3. TOOTH STATUS / MISSING TOOTH DETECTION (Present, Missing, Severely damaged, Uncertain)
      ↓
4. IMPLANT ASSISTANCE (Potential Implant Site, Mesiodistal space, Clearances)
      ↓
5. 2D VIRTUAL IMPLANT PLANNING (Adjustable Diameter, Length, Position, Angulation)
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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent.resolve()
YOLO_MODEL_PATH = ROOT_DIR / "models" / "yolov8_teeth_fdi_best.pt"

ALL_32_FDI = [
    18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
    48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38,
]

QUADRANTS = {
    "Q1": [11, 12, 13, 14, 15, 16, 17, 18],  # Upper Right (Midline -> Distal)
    "Q2": [21, 22, 23, 24, 25, 26, 27, 28],  # Upper Left  (Midline -> Distal)
    "Q4": [41, 42, 43, 44, 45, 46, 47, 48],  # Lower Right (Midline -> Distal)
    "Q3": [31, 32, 33, 34, 35, 36, 37, 38],  # Lower Left  (Midline -> Distal)
}

CLINICAL_WARNING = (
    "AI-assisted preliminary assessment from a 2D panoramic radiograph. "
    "Measurements may be affected by image magnification and distortion. "
    "Buccolingual bone width and complete 3D anatomy cannot be reliably assessed "
    "from this image alone. Final implant planning must be performed by a qualified "
    "dental professional with appropriate clinical evaluation and imaging."
)


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToothInstance:
    instance_id: int
    fdi: int
    status: str                         # "Present", "Severely damaged / Questionable", "Uncertain"
    bbox: Tuple[int, int, int, int]     # (x1, y1, x2, y2)
    centroid: Tuple[float, float]       # (cx, cy)
    width_px: float
    height_px: float
    width_mm: float
    height_mm: float
    confidence: float                   # e.g. 0.96
    is_upper: bool
    arch_region: str
    contour: Optional[np.ndarray] = None
    clinical_note: str = ""


@dataclass
class ToothStatusRecord:
    fdi: int
    arch_region: str
    status: str                         # "Present", "Missing", "Severely damaged / Questionable", "Uncertain"
    confidence: float
    instance: Optional[ToothInstance] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    available_space_mm: Optional[float] = None
    adjacent_left_fdi: Optional[int] = None
    adjacent_right_fdi: Optional[int] = None
    clinical_reason: str = ""


@dataclass
class PotentialImplantSite:
    site_id: int
    site_fdi: int
    status_label: str                   # "Potential implant site identified"
    adjacent_left_fdi: Optional[int]
    adjacent_right_fdi: Optional[int]
    center_pos: Tuple[float, float]     # (cx, cy) in pixels
    bbox: Tuple[int, int, int, int]     # (x1, y1, x2, y2) of edentulous gap
    mesiodistal_space_mm: float
    vertical_bone_height_mm: Optional[float]
    suggested_diameter_mm: float
    suggested_length_mm: float
    current_diameter_mm: float
    current_length_mm: float
    current_angulation_deg: float
    offset_x_px: float = 0.0
    offset_y_px: float = 0.0
    clearance_left_mm: float = 1.5
    clearance_right_mm: float = 1.5
    is_upper: bool = False
    arch_region: str = ""
    confidence_level: str = "Moderate"
    clinical_considerations: List[str] = field(default_factory=list)


@dataclass
class AssistanceResult:
    image_shape: Tuple[int, int]
    px_per_mm: float
    is_calibrated: bool
    tooth_instances: List[ToothInstance]
    tooth_status_table: List[ToothStatusRecord]
    potential_implant_sites: List[PotentialImplantSite]
    arch_curves: Dict[str, np.ndarray]
    landmarks: List[dict]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 & 2 — Tooth Instance Segmentation & FDI Numbering
# ─────────────────────────────────────────────────────────────────────────────

def _arch_name(fdi: int) -> str:
    q = fdi // 10
    names = {
        1: "Maxillary Right (Q1)",
        2: "Maxillary Left (Q2)",
        3: "Mandibular Left (Q3)",
        4: "Mandibular Right (Q4)",
    }
    return names.get(q, "Unknown")


def run_tooth_segmentation_and_numbering(
    img_bgr: np.ndarray,
    binary_mask: Optional[np.ndarray] = None,
    opg_width_mm: float = 150.0,
    conf_thresh: float = 0.25,
) -> Tuple[List[ToothInstance], Dict[str, np.ndarray]]:
    """
    Identifies individual tooth instances with:
    - Mask contour
    - Bounding box (x1, y1, x2, y2)
    - Centroid (cx, cy)
    - Width & Height (pixels and approximate millimeters)
    - Confidence score
    - Confident FDI numbering
    """
    h, w = img_bgr.shape[:2]
    px_per_mm = w / float(opg_width_mm)

    instances: List[ToothInstance] = []
    arch_curves: Dict[str, np.ndarray] = {}

    from utils.yolo_tooth_detector import YoloToothDetector
    yolo = YoloToothDetector()

    if yolo.is_available():
        yolo_boxes = yolo.predict(img_bgr, conf_thresh=conf_thresh)
        inst_id = 1

        for tb in yolo_boxes:
            x1, y1, x2, y2 = tb.bbox
            bw_px, bh_px = float(x2 - x1), float(y2 - y1)
            bw_mm = round(bw_px / px_per_mm, 1)
            bh_mm = round(bh_px / px_per_mm, 1)

            # Extract instance contour if binary mask is provided
            contour = None
            if binary_mask is not None:
                tooth_crop = np.zeros_like(binary_mask)
                tooth_crop[y1:y2, x1:x2] = binary_mask[y1:y2, x1:x2]
                cnts, _ = cv2.findContours(tooth_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    contour = max(cnts, key=cv2.contourArea)

            # Identify possibly damaged / questionable teeth based on coronal loss
            status = "Present"
            note = f"Tooth FDI {tb.fdi} present with intact anatomical dimensions (~{bw_mm}×{bh_mm} mm)."
            if bh_mm < 7.5:
                status = "Severely damaged / Questionable"
                note = f"Tooth FDI {tb.fdi} appears significantly compromised on panoramic imaging (severe coronal structural loss or root fragment). Professional evaluation required."

            instances.append(ToothInstance(
                instance_id=inst_id,
                fdi=tb.fdi,
                status=status,
                bbox=tb.bbox,
                centroid=(tb.cx, tb.cy),
                width_px=bw_px,
                height_px=bh_px,
                width_mm=bw_mm,
                height_mm=bh_mm,
                confidence=tb.confidence,
                is_upper=tb.is_upper,
                arch_region=tb.arch_region,
                contour=contour,
                clinical_note=note,
            ))
            inst_id += 1

    # Fit arch curves
    upper_inst = [t for t in instances if t.is_upper]
    lower_inst = [t for t in instances if not t.is_upper]

    if len(upper_inst) >= 3:
        xs_u = np.array([t.centroid[0] for t in upper_inst])
        ys_u = np.array([t.centroid[1] for t in upper_inst])
        poly_u = np.polyfit(xs_u, ys_u, 2)
        cx_eval = np.linspace(0, w, 200)
        arch_curves["upper"] = np.column_stack([cx_eval, np.polyval(poly_u, cx_eval)])

    if len(lower_inst) >= 3:
        xs_l = np.array([t.centroid[0] for t in lower_inst])
        ys_l = np.array([t.centroid[1] for t in lower_inst])
        poly_l = np.polyfit(xs_l, ys_l, 2)
        cx_eval = np.linspace(0, w, 200)
        arch_curves["lower"] = np.column_stack([cx_eval, np.polyval(poly_l, cx_eval)])

    instances.sort(key=lambda t: t.fdi)
    return instances, arch_curves


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 & 4 — 32-Tooth Status & Missing Tooth Edentulous Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_tooth_status_and_implant_sites(
    instances: List[ToothInstance],
    img_shape: Tuple[int, int],
    px_per_mm: float,
    user_params: Optional[Dict[int, dict]] = None,
) -> Tuple[List[ToothStatusRecord], List[PotentialImplantSite]]:
    user_params = user_params or {}
    h, w = img_shape
    inst_by_fdi = {t.fdi: t for t in instances}

    status_records: List[ToothStatusRecord] = []
    potential_sites: List[PotentialImplantSite] = []
    site_counter = 1

    # Analyze each quadrant along anatomical sequence
    for q_name, seq in QUADRANTS.items():
        is_upper = (q_name in ("Q1", "Q2"))
        active_missing_run: List[int] = []
        last_present: Optional[ToothInstance] = None

        for idx, fdi in enumerate(seq):
            if fdi in inst_by_fdi:
                curr_inst = inst_by_fdi[fdi]

                # Check if we just completed an edentulous gap bounded by last_present and curr_inst
                if active_missing_run and last_present is not None:
                    # Bounding coordinates of gap between adjacent teeth
                    x_left = min(last_present.bbox[2], curr_inst.bbox[0])
                    x_right = max(last_present.bbox[2], curr_inst.bbox[0])
                    gap_w_px = max(x_right - x_left, int(3.5 * px_per_mm))
                    gap_w_mm = round(gap_w_px / px_per_mm, 1)

                    center_x = (x_left + x_right) / 2.0
                    center_y = (last_present.centroid[1] + curr_inst.centroid[1]) / 2.0

                    # Preliminary suggested implant sizing
                    if gap_w_mm < 6.5:
                        sugg_diam, sugg_len = 3.3, 10.0
                    elif gap_w_mm < 9.0:
                        sugg_diam, sugg_len = 3.8, 10.0
                    elif gap_w_mm < 12.0:
                        sugg_diam, sugg_len = 4.3, 11.5
                    else:
                        sugg_diam, sugg_len = 5.0, 11.5

                    site_fdi = active_missing_run[0]
                    p = user_params.get(site_fdi, {})

                    diam_mm = float(p.get("diameter_mm", sugg_diam))
                    len_mm  = float(p.get("length_mm",   sugg_len))
                    ang_deg = float(p.get("angulation_deg", 0.0))
                    off_x   = float(p.get("offset_x_px", 0.0))
                    off_y   = float(p.get("offset_y_px", 0.0))

                    # Calculate approximate 2D clearance to left and right teeth
                    implant_radius_mm = diam_mm / 2.0
                    clearance_l = max(0.0, round((gap_w_mm / 2.0) - implant_radius_mm, 1))
                    clearance_r = max(0.0, round((gap_w_mm / 2.0) - implant_radius_mm, 1))

                    reasons = [
                        f"Potential implant site identified for FDI {site_fdi} in {_arch_name(site_fdi)}.",
                        f"Available mesiodistal space estimated at ~{gap_w_mm:.1f} mm (approximate 2D measurement).",
                        f"Approximate clearance to adjacent teeth: Left (FDI {last_present.fdi}) ~{clearance_l:.1f} mm | Right (FDI {curr_inst.fdi}) ~{clearance_r:.1f} mm.",
                    ]
                    if clearance_l < 1.5 or clearance_r < 1.5:
                        reasons.append("⚠ Limited interdental clearance (<1.5 mm). Narrow platform or orthodontic space opening may be considered.")
                    reasons.append("Mandatory requirement: Buccolingual ridge width and bone density must be evaluated on 3D CBCT.")

                    potential_sites.append(PotentialImplantSite(
                        site_id=site_counter,
                        site_fdi=site_fdi,
                        status_label="Potential implant site identified",
                        adjacent_left_fdi=last_present.fdi,
                        adjacent_right_fdi=curr_inst.fdi,
                        center_pos=(center_x, center_y),
                        bbox=(int(x_left), int(center_y - 25), int(x_right), int(center_y + 25)),
                        mesiodistal_space_mm=gap_w_mm,
                        vertical_bone_height_mm=None,
                        suggested_diameter_mm=sugg_diam,
                        suggested_length_mm=sugg_len,
                        current_diameter_mm=diam_mm,
                        current_length_mm=len_mm,
                        current_angulation_deg=ang_deg,
                        offset_x_px=off_x,
                        offset_y_px=off_y,
                        clearance_left_mm=clearance_l,
                        clearance_right_mm=clearance_r,
                        is_upper=is_upper,
                        arch_region=_arch_name(site_fdi),
                        confidence_level="Moderate (2D Projection)",
                        clinical_considerations=reasons,
                    ))
                    site_counter += 1

                active_missing_run = []
                last_present = curr_inst

                status_records.append(ToothStatusRecord(
                    fdi=fdi,
                    arch_region=curr_inst.arch_region,
                    status=curr_inst.status,
                    confidence=curr_inst.confidence,
                    instance=curr_inst,
                    bbox=curr_inst.bbox,
                    available_space_mm=curr_inst.width_mm,
                    clinical_reason=curr_inst.clinical_note,
                ))

            else:
                active_missing_run.append(fdi)

                if fdi in (18, 28, 38, 48):
                    status_records.append(ToothStatusRecord(
                        fdi=fdi,
                        arch_region=_arch_name(fdi),
                        status="Uncertain",
                        confidence=0.70,
                        clinical_reason="Third molar not observed in dental arch (absent, unerupted, or previous extraction).",
                    ))
                else:
                    status_records.append(ToothStatusRecord(
                        fdi=fdi,
                        arch_region=_arch_name(fdi),
                        status="Missing",
                        confidence=0.92,
                        clinical_reason=f"FDI {fdi} missing in anatomical dental sequence. Edentulous space detected; implant replacement may be considered.",
                    ))

    status_map = {r.fdi: r for r in status_records}
    ordered_records = [status_map[fdi] for fdi in ALL_32_FDI if fdi in status_map]

    return ordered_records, potential_sites


# ─────────────────────────────────────────────────────────────────────────────
# Visual Overview Renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_assistance_figure(
    img_bgr: np.ndarray,
    instances: List[ToothInstance],
    sites: List[PotentialImplantSite],
    arch_curves: Dict[str, np.ndarray],
    px_per_mm: float,
    show_segmentation: bool = True,
    show_boxes: bool = True,
    show_fdi: bool = True,
    show_missing: bool = True,
    show_implants: bool = True,
    show_measurements: bool = True,
    show_clearance: bool = True,
    figsize: Tuple[int, int] = (22, 9),
) -> plt.Figure:
    h, w = img_bgr.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb, aspect="auto", alpha=0.92)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    # 1. Individual Tooth Masks and Bounding Boxes
    for t in instances:
        x1, y1, x2, y2 = t.bbox
        ec = "#2ECC71" if t.status == "Present" else "#F39C12"

        if show_boxes:
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.6, edgecolor=ec, facecolor=ec, alpha=0.12, zorder=3,
            )
            ax.add_patch(rect)

        if show_segmentation and t.contour is not None:
            poly_pts = t.contour.reshape(-1, 2)
            if len(poly_pts) >= 3:
                cnt_patch = mpatches.Polygon(
                    poly_pts, closed=True, facecolor=ec, edgecolor=ec,
                    alpha=0.30, linewidth=1.2, zorder=4,
                )
                ax.add_patch(cnt_patch)

        if show_fdi:
            label_y = y1 - 8 if t.is_upper else y2 + 16
            tag = f"{t.fdi} ✓" if t.status == "Present" else f"{t.fdi} ⚠"
            ax.text(
                t.centroid[0], label_y, tag,
                color="#F1C40F" if t.status == "Present" else "#F39C12",
                fontsize=8.0, ha="center", va="center", fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#101820", alpha=0.85, edgecolor="none"),
            )

    # 2. Missing Tooth / Edentulous Gaps
    if show_missing:
        for site in sites:
            gx1, gy1, gx2, gy2 = site.bbox
            rect_gap = mpatches.Rectangle(
                (gx1, gy1), gx2 - gx1, gy2 - gy1,
                linewidth=2.0, edgecolor="#E74C3C", facecolor="#E74C3C", alpha=0.16,
                linestyle="--", zorder=4,
            )
            ax.add_patch(rect_gap)

            label_gy = gy1 - 12 if site.is_upper else gy2 + 18
            ax.text(
                site.center_pos[0], label_gy,
                f"POTENTIAL IMPLANT SITE\nFDI {site.site_fdi} (MISSING)",
                color="#FF6B6B", fontsize=8.0, ha="center", va="center",
                fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#1A0000", alpha=0.90, edgecolor="#E74C3C", lw=1.2),
            )

    # 3. Mesiodistal Measurements & Clearances
    if show_measurements:
        for site in sites:
            gx1, _, gx2, _ = site.bbox
            my = site.center_pos[1]
            ax.plot([gx1, gx2], [my, my], color="#F39C12", lw=1.8, ls="-", zorder=5)
            ax.plot([gx1, gx1], [my - 6, my + 6], color="#F39C12", lw=1.8, zorder=5)
            ax.plot([gx2, gx2], [my - 6, my + 6], color="#F39C12", lw=1.8, zorder=5)

            ax.text(
                site.center_pos[0], my - 8,
                f"~{site.mesiodistal_space_mm:.1f} mm (approx)",
                color="#F39C12", fontsize=7.5, ha="center", va="bottom",
                fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#111118", alpha=0.80, edgecolor="none"),
            )

    # 4. Virtual Implant Overlay
    if show_implants:
        for site in sites:
            cx = site.center_pos[0] + site.offset_x_px
            cy = site.center_pos[1] + site.offset_y_px
            diam_px = site.current_diameter_mm * px_per_mm
            len_px  = site.current_length_mm * px_per_mm
            ang_deg = site.current_angulation_deg

            from utils.panoramic_planner import _implant_polygon
            poly = _implant_polygon(cx, cy, diam_px, len_px, ang_deg, site.is_upper)
            patch = mpatches.Polygon(
                poly, closed=True,
                facecolor="#00E5FF", alpha=0.45,
                edgecolor="#00E5FF", linewidth=2.2, zorder=7,
            )
            ax.add_patch(patch)

            # Centerline
            d = -1 if site.is_upper else 1
            ang_rad = np.radians(ang_deg)
            c_, s_ = np.cos(ang_rad), np.sin(ang_rad)
            ax.plot([cx, cx + s_ * len_px], [cy, cy + c_ * d * len_px], color="#FFFFFF", lw=1.4, ls=":", zorder=8)

            # Badge
            badge_y = cy - 22 if site.is_upper else cy + 22
            ax.text(
                cx, badge_y,
                f"Virtual Implant FDI {site.site_fdi}\nØ{site.current_diameter_mm:.1f} × {site.current_length_mm:.0f} mm",
                color="#00E5FF", fontsize=7.5, ha="center", va="center",
                fontweight="bold", zorder=9,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#001820", alpha=0.90, edgecolor="#00E5FF", lw=1.0),
            )

    # Legend
    handles = [
        mpatches.Patch(color="#2ECC71", label="Individual Tooth Instance (Present)"),
        mpatches.Patch(color="#E74C3C", label="Potential Implant Site (Missing Gap)"),
        mpatches.Patch(color="#F39C12", label="Severely Damaged / Questionable Tooth (⚠)"),
        mpatches.Patch(color="#00E5FF", label="2D Virtual Implant Overlay (Adjustable)"),
    ]
    ax.legend(
        handles=handles, loc="lower left", fontsize=8,
        facecolor="#0d1117", edgecolor="#334155", labelcolor="white", framealpha=0.85,
    )

    fig.text(
        0.5, 0.01,
        "⚠ CLINICAL DISCLAIMER: AI-assisted preliminary assessment from a 2D panoramic radiograph. CBCT and professional evaluation required before surgery.",
        color="#94A3B8", fontsize=8.0, ha="center", va="bottom", style="italic",
    )
    fig.tight_layout(pad=0.5)
    return fig
