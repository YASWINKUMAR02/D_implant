"""
implant_suggester_3d.py
=======================
AI Dental Implant Suggestion Engine for 3D CBCT volumes (OralSeg 35-class segmentation).

Given the OralSeg segmentation volume and seg_info dict, this module:
  1. Identifies all FDI tooth sites that are missing / not detected.
  2. Locates the 3D voxel coordinates of each missing site.
  3. Measures alveolar bone height, ridge width, and mandibular canal
     clearance at each site using the existing bone_measurements utilities.
  4. Selects the optimal implant diameter/length from a clinical size database
     that fits within the available bone, with appropriate safety margins.
  5. Scores each site with a 0–1 feasibility index.
  6. Returns a ranked list of ImplantSuggestion3D objects.
  7. Provides helper functions for HTML card generation and session-state caching.

Architecture note:
  This module deliberately avoids import of Streamlit so it can be used in
  offline / scripting contexts. Streamlit rendering is handled in app.py.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Clinical Implant Size Database
# Each entry: (diameter_mm, length_mm, min_bone_height_mm, min_ridge_width_mm, label)
# Ordered from smallest to largest — selection picks first entry that fits.
# ─────────────────────────────────────────────────────────────────────────────
_IMPLANT_DB: List[Tuple[float, float, float, float, str]] = [
    # diameter, length, min_bone_h, min_ridge_w, label
    (3.3,  8.0,   8.0,  5.5,  "Ø3.3 × 8 mm   — Narrow / Short"),
    (3.3, 10.0,  10.5,  5.5,  "Ø3.3 × 10 mm  — Narrow Platform"),
    (3.5, 10.0,  11.0,  5.5,  "Ø3.5 × 10 mm  — Narrow Platform"),
    (3.5, 11.5,  12.5,  5.5,  "Ø3.5 × 11.5 mm — Narrow Platform"),
    (4.0,  8.0,   8.5,  6.5,  "Ø4.0 × 8 mm   — Regular / Short"),
    (4.0, 10.0,  10.5,  6.5,  "Ø4.0 × 10 mm  — Regular Platform"),
    (4.0, 11.5,  12.0,  6.5,  "Ø4.0 × 11.5 mm — Regular Platform"),
    (4.0, 13.0,  13.5,  6.5,  "Ø4.0 × 13 mm  — Regular Platform"),
    (4.5, 10.0,  11.0,  7.0,  "Ø4.5 × 10 mm  — Regular Wide"),
    (4.5, 11.5,  12.5,  7.0,  "Ø4.5 × 11.5 mm — Regular Wide"),
    (5.0, 10.0,  11.5,  8.0,  "Ø5.0 × 10 mm  — Wide Platform"),
    (5.0, 11.5,  12.5,  8.0,  "Ø5.0 × 11.5 mm — Wide Platform"),
    (5.5, 10.0,  11.5,  9.5,  "Ø5.5 × 10 mm  — Wide Platform"),
    (6.0, 10.0,  12.0, 10.0,  "Ø6.0 × 10 mm  — Extra-Wide Platform"),
]

# Minimum canal safety clearance (mm) — standard clinical recommendation
CANAL_SAFETY_MM = 2.0
# Bone buffer added to implant length (apex safety margin, mm)
APEX_SAFETY_MM = 1.5

# Feasibility thresholds
EXCELLENT_THRESHOLD = 0.78
ACCEPTABLE_THRESHOLD = 0.55


@dataclass
class ImplantSuggestion3D:
    """Represents a single 3D CBCT implant site suggestion."""
    rank: int
    fdi_tooth: int
    is_mandibular: bool
    arch_region: str               # "Maxillary Right" etc.

    # Bone measurements
    bone_height_mm: float
    ridge_width_mm: float
    canal_distance_mm: Optional[float]  # None for maxillary sites

    # Suggested implant
    recommended_diameter_mm: float
    recommended_length_mm: float
    implant_label: str

    # Scoring
    feasibility_score: float       # 0–1
    feasibility_label: str         # "Excellent", "Acceptable", "Caution"
    feasibility_emoji: str         # 🟢 / 🟡 / 🔴

    # Optional spatial reference
    site_center_vox: Optional[np.ndarray] = None

    # Notes / warnings
    notes: List[str] = field(default_factory=list)


def _fdi_to_arch_region(fdi: int) -> Tuple[str, bool]:
    """Map FDI tooth number to clinical arch region string and mandibular flag."""
    q = fdi // 10
    t = fdi % 10
    is_mandibular = q in (3, 4)
    jaw = "Mandibular" if is_mandibular else "Maxillary"
    side_map = {1: "Right", 2: "Left", 3: "Left", 4: "Right"}
    side = side_map.get(q, "?")
    region_names = {
        1: "Molar" if t >= 6 else ("Premolar" if t >= 4 else ("Canine" if t == 3 else "Incisor")),
        2: "Molar" if t >= 6 else ("Premolar" if t >= 4 else ("Canine" if t == 3 else "Incisor")),
        3: "Molar" if t >= 6 else ("Premolar" if t >= 4 else ("Canine" if t == 3 else "Incisor")),
        4: "Molar" if t >= 6 else ("Premolar" if t >= 4 else ("Canine" if t == 3 else "Incisor")),
    }
    tooth_type = region_names.get(q, "Tooth")
    return f"{jaw} {side} {tooth_type} (FDI {fdi})", is_mandibular


def _select_implant(
    bone_height_mm: float,
    ridge_width_mm: float,
    canal_dist_mm: Optional[float],
) -> Tuple[float, float, str]:
    """
    Select the best-fitting candidate implant baseline from the clinical database.
    """
    available_height = bone_height_mm - APEX_SAFETY_MM
    available_width  = ridge_width_mm

    if canal_dist_mm is not None:
        available_height = min(available_height, bone_height_mm - CANAL_SAFETY_MM)

    best_diam, best_len, best_label = 3.3, 8.0, _IMPLANT_DB[0][4]
    for diam, length, min_h, min_w, label in _IMPLANT_DB:
        if length <= available_height and diam <= available_width - 1.0:
            best_diam, best_len, best_label = diam, length, label

    return best_diam, best_len, best_label


def suggest_preliminary_implant_range_for_site(
    bone_height_mm: float,
    ridge_width_mm: float,
    canal_dist_mm: Optional[float],
) -> Dict[str, Any]:
    """
    Estimate reasonable preliminary implant dimension ranges based on measured 3D anatomy.
    
    Returns:
        {
            "diameter_range_str": "3.5–4.5 mm",
            "length_range_str": "8–11.5 mm",
            "default_diameter_mm": 4.0,
            "default_length_mm": 10.0,
            "clinical_notes": [...]
        }
    """
    # Max safe diameter = ridge_width - 2.0 mm (1mm buccal + 1mm lingual margin)
    max_d = max(3.0, ridge_width_mm - 2.0)
    min_d = 3.0 if max_d <= 3.5 else 3.5
    d_range_str = f"{min_d:.1f}–{min(max_d, 5.5):.1f} mm"

    # Max safe length = available height - apex safety
    avail_h = bone_height_mm - APEX_SAFETY_MM
    if canal_dist_mm is not None:
        avail_h = min(avail_h, bone_height_mm - CANAL_SAFETY_MM)
    
    max_l = max(6.0, min(avail_h, 14.0))
    min_l = 6.0 if max_l <= 8.0 else 8.0
    l_range_str = f"{min_l:.0f}–{max_l:.0f} mm"

    default_d, default_l, _ = _select_implant(bone_height_mm, ridge_width_mm, canal_dist_mm)

    notes = []
    if ridge_width_mm < 6.0:
        notes.append("Narrow alveolar ridge — narrow-platform implant (Ø3.0–3.5 mm) or ridge augmentation indicated.")
    if canal_dist_mm is not None and canal_dist_mm < 2.5:
        notes.append("Close proximity to mandibular canal — conservative implant length recommended.")

    return {
        "diameter_range_str": d_range_str,
        "length_range_str": l_range_str,
        "default_diameter_mm": default_d,
        "default_length_mm": default_l,
        "notes": notes,
    }


def _compute_feasibility_score(
    bone_height_mm: float,
    ridge_width_mm: float,
    canal_dist_mm: Optional[float],
    implant_diam: float,
    implant_len: float,
) -> Tuple[float, str, str]:
    """
    Compute a 0–1 feasibility score for the site.

    Returns:
        (score, label, emoji)
    """
    # --- Bone Height Score ---
    # Minimum safe bone height = implant length + apex safety margin
    min_required_h = implant_len + APEX_SAFETY_MM
    if bone_height_mm <= 0:
        h_score = 0.0
    else:
        h_score = float(np.clip(bone_height_mm / min_required_h, 0.0, 1.0))

    # --- Ridge Width Score ---
    # Need implant diameter + 1 mm buccal + 1 mm lingual = Ø + 2 mm
    min_required_w = implant_diam + 2.0
    if ridge_width_mm <= 0:
        w_score = 0.0
    else:
        w_score = float(np.clip(ridge_width_mm / min_required_w, 0.0, 1.0))
        # Bonus for particularly wide ridges (above 2× minimum)
        if ridge_width_mm >= min_required_w * 1.5:
            w_score = min(w_score * 1.1, 1.0)

    # --- Canal Safety Score ---
    if canal_dist_mm is None:
        # Maxillary site — no inferior alveolar nerve risk
        c_score = 1.0
    elif canal_dist_mm >= CANAL_SAFETY_MM:
        # Linear bonus for clearance above threshold, capped at 3× threshold
        c_score = float(np.clip(canal_dist_mm / (CANAL_SAFETY_MM * 3), 0.5, 1.0))
    else:
        # Below minimum — penalize proportionally
        c_score = float(np.clip(canal_dist_mm / CANAL_SAFETY_MM, 0.0, 0.49))

    # Weighted composite
    score = 0.35 * h_score + 0.30 * w_score + 0.35 * c_score
    score = float(np.clip(score, 0.0, 1.0))

    if score >= EXCELLENT_THRESHOLD:
        label, emoji = "Excellent", "🟢"
    elif score >= ACCEPTABLE_THRESHOLD:
        label, emoji = "Acceptable", "🟡"
    else:
        label, emoji = "Caution", "🔴"

    return round(score, 3), label, emoji


def _build_notes(
    bone_height_mm: float,
    ridge_width_mm: float,
    canal_dist_mm: Optional[float],
    implant_len: float,
    implant_diam: float,
) -> List[str]:
    """Generate clinical notes / warnings for the suggestion."""
    notes = []
    min_h = implant_len + APEX_SAFETY_MM
    if bone_height_mm < min_h:
        notes.append(
            f"⚠ Bone height ({bone_height_mm:.1f} mm) is below minimum recommended "
            f"({min_h:.1f} mm). Bone augmentation may be required."
        )
    min_w = implant_diam + 2.0
    if ridge_width_mm < min_w:
        notes.append(
            f"⚠ Ridge width ({ridge_width_mm:.1f} mm) is narrow for Ø{implant_diam} mm implant "
            f"(minimum {min_w:.1f} mm). Guided bone regeneration may be needed."
        )
    if canal_dist_mm is not None and canal_dist_mm < CANAL_SAFETY_MM:
        notes.append(
            f"🚨 Canal clearance ({canal_dist_mm:.1f} mm) is below the {CANAL_SAFETY_MM} mm "
            f"safety threshold. Risk of inferior alveolar nerve injury — surgical guide mandatory."
        )
    elif canal_dist_mm is not None and canal_dist_mm < CANAL_SAFETY_MM * 1.5:
        notes.append(
            f"⚠ Canal clearance ({canal_dist_mm:.1f} mm) is marginal. Surgical guide strongly recommended."
        )
    if not notes:
        notes.append("✅ Site appears suitable for standard implant placement.")
    return notes


def auto_suggest_implants_3d(
    seg_vol: np.ndarray,
    seg_info: Dict,
    voxel_spacing_mm: List[float] = None,
    max_suggestions: int = 10,
    min_feasibility: float = 0.30,
) -> List[ImplantSuggestion3D]:
    """
    Automatically identify and rank all viable implant sites from a 3D CBCT
    OralSeg segmentation.

    Args:
        seg_vol: uint8 3D segmentation volume (x, y, z) with 0–35 label values.
        seg_info: dict from oralseg_inference.analyze_segmentation().
        voxel_spacing_mm: [dx, dy, dz] physical voxel size in mm. Defaults to [1,1,1].
        max_suggestions: cap on number of returned suggestions.
        min_feasibility: skip sites below this score (reduces computation).

    Returns:
        Ranked list of ImplantSuggestion3D (best first).
    """
    from utils.implant_geometry import (
        detect_potential_missing_teeth, get_tooth_site_coordinates,
    )
    from utils.bone_measurements import get_site_measurements
    from utils.collision_analysis import analyze_implant_safety
    from utils.implant_geometry import sample_implant_cylinder_points

    if voxel_spacing_mm is None or len(voxel_spacing_mm) < 3:
        voxel_spacing_mm = [1.0, 1.0, 1.0]

    potential_missing = detect_potential_missing_teeth(seg_info)
    logger.info(f"Auto-suggesting implants for {len(potential_missing)} missing/not-detected teeth: {potential_missing}")

    if not potential_missing:
        logger.info("No missing teeth detected — no implant suggestions generated.")
        return []

    raw_suggestions = []

    for fdi in potential_missing:
        try:
            # Locate site
            center_vox, is_detected = get_tooth_site_coordinates(
                seg_vol, fdi, voxel_spacing_mm=voxel_spacing_mm
            )

            # Bone measurements
            site_meas = get_site_measurements(
                seg_vol, center_vox, fdi, voxel_spacing_mm=voxel_spacing_mm
            )
            bone_h = site_meas["bone_height_mm"]
            ridge_w = site_meas["ridge_width_mm"]
            is_mandibular = site_meas["is_mandibular"]

            # Canal distance (for mandibular sites only)
            canal_dist: Optional[float] = None
            if is_mandibular:
                try:
                    implant_pts = sample_implant_cylinder_points(
                        center_vox, diameter_mm=4.0, length_mm=10.0,
                        voxel_spacing_mm=voxel_spacing_mm,
                    )
                    safety_res = analyze_implant_safety(
                        implant_pts, seg_vol, fdi,
                        planning_threshold_mm=CANAL_SAFETY_MM,
                        voxel_spacing_mm=voxel_spacing_mm,
                    )
                    canal_dist = safety_res.get("canal_distance_mm")
                except Exception as ce:
                    logger.warning(f"Canal analysis failed for FDI {fdi}: {ce}")
                    canal_dist = None

            # Select implant
            diam, length, imp_label = _select_implant(bone_h, ridge_w, canal_dist)

            # Score
            score, feat_label, emoji = _compute_feasibility_score(
                bone_h, ridge_w, canal_dist, diam, length
            )

            if score < min_feasibility:
                logger.debug(f"FDI {fdi} skipped (score {score:.2f} < {min_feasibility})")
                continue

            arch_region, _ = _fdi_to_arch_region(fdi)
            notes = _build_notes(bone_h, ridge_w, canal_dist, length, diam)

            raw_suggestions.append(ImplantSuggestion3D(
                rank=0,  # set after sorting
                fdi_tooth=fdi,
                is_mandibular=is_mandibular,
                arch_region=arch_region,
                bone_height_mm=bone_h,
                ridge_width_mm=ridge_w,
                canal_distance_mm=canal_dist,
                recommended_diameter_mm=diam,
                recommended_length_mm=length,
                implant_label=imp_label,
                feasibility_score=score,
                feasibility_label=feat_label,
                feasibility_emoji=emoji,
                site_center_vox=center_vox,
                notes=notes,
            ))

        except Exception as e:
            logger.warning(f"Could not generate suggestion for FDI {fdi}: {e}")
            continue

    # Rank by feasibility score descending
    raw_suggestions.sort(key=lambda s: -s.feasibility_score)
    ranked = raw_suggestions[:max_suggestions]
    for i, s in enumerate(ranked):
        s.rank = i + 1

    logger.info(f"Generated {len(ranked)} implant suggestions.")
    return ranked


def seg_vol_cache_key(seg_vol: np.ndarray, case_name: str) -> str:
    """
    Compute a fast cache key for the segmentation volume.
    Uses case name + shape + a checksum of a small subsample to avoid
    hashing the full volume (which can be large).
    """
    shape_str = "x".join(str(d) for d in seg_vol.shape)
    # Sample every 50th element for fast checksum
    sample = seg_vol.flat[::50].tobytes()
    digest = hashlib.md5(sample).hexdigest()[:12]
    return f"{case_name}_{shape_str}_{digest}"


def suggestions_to_dict_list(suggestions: List[ImplantSuggestion3D]) -> List[dict]:
    """Convert 3D suggestion list to JSON-serialisable list of dicts for export."""
    out = []
    for s in suggestions:
        out.append({
            "rank":                     s.rank,
            "fdi_tooth":                s.fdi_tooth,
            "arch_region":              s.arch_region,
            "bone_height_mm":           s.bone_height_mm,
            "ridge_width_mm":           s.ridge_width_mm,
            "canal_distance_mm":        s.canal_distance_mm,
            "recommended_diameter_mm":  s.recommended_diameter_mm,
            "recommended_length_mm":    s.recommended_length_mm,
            "implant_label":            s.implant_label,
            "feasibility_score":        s.feasibility_score,
            "feasibility_label":        s.feasibility_label,
            "notes":                    s.notes,
        })
    return out


def build_suggestion_summary_html(suggestions: List[ImplantSuggestion3D]) -> str:
    """
    Generate a styled HTML table summarising all 3D suggestions.
    Suitable for embedding directly in a Streamlit st.markdown(unsafe_allow_html=True).
    """
    if not suggestions:
        return (
            "<div style='padding:12px;background:#FEF2F2;border-radius:8px;"
            "border:1px solid #FCA5A5;color:#7F1D1D;font-size:12.5px;'>"
            "No viable implant sites identified. All detected teeth are present, "
            "or bone measurements indicate insufficient bone at all missing sites."
            "</div>"
        )

    rows = ""
    for s in suggestions:
        canal_str = f"{s.canal_distance_mm:.1f} mm" if s.canal_distance_mm is not None else "N/A (Maxilla)"
        score_pct = int(s.feasibility_score * 100)

        # Progress bar color
        if s.feasibility_score >= EXCELLENT_THRESHOLD:
            bar_color = "#16A34A"
        elif s.feasibility_score >= ACCEPTABLE_THRESHOLD:
            bar_color = "#D97706"
        else:
            bar_color = "#DC2626"

        score_bar = (
            f"<div style='background:#E2E8F0;border-radius:4px;height:6px;margin-top:3px;'>"
            f"<div style='background:{bar_color};width:{score_pct}%;height:6px;border-radius:4px;'></div>"
            f"</div>"
        )

        rows += f"""
        <tr>
          <td style='text-align:center;font-weight:700;color:#0F3B7A;'>{s.rank}</td>
          <td style='font-weight:700;'>{s.fdi_tooth}</td>
          <td style='font-size:11px;color:#334155;'>{s.arch_region}</td>
          <td style='text-align:right;font-family:monospace;'>{s.bone_height_mm:.1f}</td>
          <td style='text-align:right;font-family:monospace;'>{s.ridge_width_mm:.1f}</td>
          <td style='text-align:right;font-family:monospace;'>{canal_str}</td>
          <td style='font-size:11px;font-weight:600;color:#1E40AF;'>Ø{s.recommended_diameter_mm} × {s.recommended_length_mm} mm</td>
          <td>
            <span style='font-size:12px;'>{s.feasibility_emoji}</span>
            <span style='font-size:11px;font-weight:600;'>{s.feasibility_label}</span>
            {score_bar}
          </td>
        </tr>"""

    html = f"""
    <div style='overflow-x:auto;'>
    <table style='width:100%;border-collapse:collapse;font-size:12px;font-family:Inter,sans-serif;'>
      <thead>
        <tr style='background:#0F3B7A;color:#FFFFFF;'>
          <th style='padding:8px 6px;text-align:center;'>Rank</th>
          <th style='padding:8px 6px;'>FDI</th>
          <th style='padding:8px 6px;'>Arch Region</th>
          <th style='padding:8px 6px;text-align:right;'>Bone H (mm)</th>
          <th style='padding:8px 6px;text-align:right;'>Ridge W (mm)</th>
          <th style='padding:8px 6px;text-align:right;'>Canal Dist</th>
          <th style='padding:8px 6px;'>Recommended Implant</th>
          <th style='padding:8px 6px;'>Feasibility</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
    </div>
    """
    return html
