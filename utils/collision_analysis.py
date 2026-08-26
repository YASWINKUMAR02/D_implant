"""
collision_analysis.py
Performs comprehensive 3D safety clearance and collision checking:
  - Implant ↔ Mandibular Canal clearance (mm)
  - Implant ↔ Mesial adjacent tooth root clearance (mm)
  - Implant ↔ Distal adjacent tooth root clearance (mm)
  - Implant ↔ Buccal & Lingual cortical plates (mm)
  - Cortical bone breach / containment status
All calculations use true physical millimeter coordinates derived from voxel spacing.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from utils.oralseg_inference import LABEL_MAP, TOOTH_FDI_MAP
from utils.implant_geometry import FDI_TO_LABEL, ALL_FDI_TEETH
from utils.bone_measurements import calculate_adjacent_root_distances, calculate_cortical_plate_clearances


def calculate_canal_clearance_distance(
    implant_points_voxel: np.ndarray,
    seg_vol: np.ndarray,
    voxel_spacing_mm: List[float] = [1.0, 1.0, 1.0],
    search_margin_vox: int = 40,
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """
    Calculate minimum 3D Euclidean distance (in mm) between the virtual implant 
    and the Mandibular Canal (Label 35).
    
    Returns:
        (min_distance_mm, nearest_canal_point_vox)
    """
    canal_label = 35
    
    min_pt = np.floor(implant_points_voxel.min(axis=0)).astype(int) - search_margin_vox
    max_pt = np.ceil(implant_points_voxel.max(axis=0)).astype(int) + search_margin_vox

    min_pt = np.clip(min_pt, 0, np.array(seg_vol.shape) - 1)
    max_pt = np.clip(max_pt, 0, np.array(seg_vol.shape) - 1)

    sub_vol = seg_vol[min_pt[0]:max_pt[0]+1, min_pt[1]:max_pt[1]+1, min_pt[2]:max_pt[2]+1]
    local_canal_vox = np.argwhere(sub_vol == canal_label)

    if len(local_canal_vox) == 0:
        global_canal_vox = np.argwhere(seg_vol == canal_label)
        if len(global_canal_vox) == 0:
            return None, None
        step = max(1, len(global_canal_vox) // 500)
        canal_sample_vox = global_canal_vox[::step]
    else:
        canal_sample_vox = local_canal_vox + min_pt

    spacing = np.array(voxel_spacing_mm)
    implant_pts_mm = implant_points_voxel * spacing
    canal_pts_mm = canal_sample_vox * spacing

    step_imp = max(1, len(implant_pts_mm) // 60)
    imp_sub = implant_pts_mm[::step_imp]

    diff = imp_sub[:, np.newaxis, :] - canal_pts_mm[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)

    min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
    min_dist_mm = float(dist_matrix[min_idx])
    nearest_canal_vox = canal_sample_vox[min_idx[1]]

    return round(min_dist_mm, 1), nearest_canal_vox


def check_implant_collisions(
    implant_points_voxel: np.ndarray,
    seg_vol: np.ndarray,
    selected_fdi_tooth: int,
) -> Dict[str, Any]:
    """
    Check if the virtual implant intersects adjacent teeth or mandibular canal.
    """
    valid_coords = []
    for pt in implant_points_voxel:
        ix, iy, iz = int(round(pt[0])), int(round(pt[1])), int(round(pt[2]))
        if 0 <= ix < seg_vol.shape[0] and 0 <= iy < seg_vol.shape[1] and 0 <= iz < seg_vol.shape[2]:
            valid_coords.append([ix, iy, iz])

    if not valid_coords:
        return {
            "has_collision": False,
            "colliding_labels": [],
            "colliding_names": [],
            "status_text": "🟢 No segmentation collision detected",
        }

    coords_arr = np.array(valid_coords)
    sampled_labels = seg_vol[coords_arr[:, 0], coords_arr[:, 1], coords_arr[:, 2]]
    unique_labels, counts = np.unique(sampled_labels, return_counts=True)

    selected_tooth_label = FDI_TO_LABEL.get(selected_fdi_tooth, -1)
    ignored_labels = {0, 1, 2, selected_tooth_label}

    colliding_labels = []
    colliding_names = []

    for lbl, count in zip(unique_labels, counts):
        if lbl not in ignored_labels and count >= 3:
            colliding_labels.append(int(lbl))
            colliding_names.append(LABEL_MAP.get(lbl, f"Structure {lbl}"))

    has_collision = len(colliding_labels) > 0

    if has_collision:
        names_str = ", ".join(colliding_names)
        status_text = f"🔴 Potential Collision Detected with {names_str}"
    else:
        status_text = "🟢 No direct structure collision detected"

    return {
        "has_collision": has_collision,
        "colliding_labels": colliding_labels,
        "colliding_names": colliding_names,
        "status_text": status_text,
    }


def analyze_implant_safety(
    implant_points_voxel: np.ndarray,
    seg_vol: np.ndarray,
    selected_fdi_tooth: int,
    detected_teeth: Optional[List[int]] = None,
    planning_threshold_mm: float = 2.0,
    root_threshold_mm: float = 1.5,
    cortical_threshold_mm: float = 1.0,
    voxel_spacing_mm: List[float] = [1.0, 1.0, 1.0],
) -> Dict[str, Any]:
    """
    Combined multi-parameter safety and collision analysis for the configured virtual implant.
    Evaluates:
      1. Mandibular canal clearance (>= 2.0 mm)
      2. Mesial root clearance (>= 1.5 mm)
      3. Distal root clearance (>= 1.5 mm)
      4. Buccal cortical plate (>= 1.0 mm)
      5. Lingual cortical plate (>= 1.0 mm)
      6. Bone containment & breach detection
    """
    if detected_teeth is None:
        detected_teeth = ALL_FDI_TEETH

    is_mandibular = (selected_fdi_tooth >= 31 and selected_fdi_tooth <= 48)

    # 1. Canal clearance
    canal_dist_mm, nearest_canal_vox = calculate_canal_clearance_distance(
        implant_points_voxel, seg_vol, voxel_spacing_mm=voxel_spacing_mm
    )

    # 2. Adjacent root clearances
    mesial_root_mm, distal_root_mm, mesial_fdi, distal_fdi = calculate_adjacent_root_distances(
        implant_points_voxel, seg_vol, selected_fdi_tooth, detected_teeth, voxel_spacing_mm=voxel_spacing_mm
    )

    # 3. Cortical plate clearances & containment
    buccal_plate_mm, lingual_plate_mm, is_contained = calculate_cortical_plate_clearances(
        implant_points_voxel, seg_vol, selected_fdi_tooth, voxel_spacing_mm=voxel_spacing_mm
    )

    # 4. Direct volumetric collision
    collision_info = check_implant_collisions(
        implant_points_voxel, seg_vol, selected_fdi_tooth=selected_fdi_tooth
    )

    # 5. Checklist building & Risk Tier Evaluation
    checklist = []
    has_critical = False
    has_caution = False

    # A. Canal check
    if is_mandibular:
        if canal_dist_mm is not None:
            if canal_dist_mm >= planning_threshold_mm:
                c_status = "PASS"
                c_icon = "✓"
                c_badge = "🟢 Favorable"
            elif canal_dist_mm >= 1.0:
                c_status = "CAUTION"
                c_icon = "⚠"
                c_badge = "🟡 Limited Clearance"
                has_caution = True
            else:
                c_status = "DANGER"
                c_icon = "✕"
                c_badge = "🔴 Critical Proximity"
                has_critical = True
            checklist.append({
                "item": "Mandibular Canal Clearance",
                "measured_mm": canal_dist_mm,
                "threshold_mm": planning_threshold_mm,
                "status": c_status,
                "icon": c_icon,
                "badge": c_badge,
                "note": f"{canal_dist_mm:.1f} mm from nerve canal (safe threshold ≥ {planning_threshold_mm:.1f} mm)",
            })
        else:
            checklist.append({
                "item": "Mandibular Canal Clearance",
                "measured_mm": None,
                "threshold_mm": planning_threshold_mm,
                "status": "UNKNOWN",
                "icon": "⚪",
                "badge": "⚪ Canal Not Segmented in ROI",
                "note": "Canal structure not detected in local quadrant",
            })
    else:
        checklist.append({
            "item": "Inferior Alveolar Canal",
            "measured_mm": None,
            "threshold_mm": None,
            "status": "N/A",
            "icon": "✓",
            "badge": "🟢 Not Applicable (Maxilla)",
            "note": "Maxillary arch site — evaluate sinus floor on coronal/panoramic views",
        })

    # B. Mesial root check
    if mesial_root_mm is not None:
        if mesial_root_mm >= root_threshold_mm:
            m_status = "PASS"
            m_icon = "✓"
            m_badge = "🟢 Favorable"
        elif mesial_root_mm >= 0.8:
            m_status = "CAUTION"
            m_icon = "⚠"
            m_badge = "🟡 Proximity Warning"
            has_caution = True
        else:
            m_status = "DANGER"
            m_icon = "✕"
            m_badge = "🔴 Root Collision Risk"
            has_critical = True
        checklist.append({
            "item": f"Mesial Root (FDI {mesial_fdi})",
            "measured_mm": mesial_root_mm,
            "threshold_mm": root_threshold_mm,
            "status": m_status,
            "icon": m_icon,
            "badge": m_badge,
            "note": f"{mesial_root_mm:.1f} mm to adjacent root (recommended ≥ {root_threshold_mm:.1f} mm)",
        })

    # C. Distal root check
    if distal_root_mm is not None:
        if distal_root_mm >= root_threshold_mm:
            d_status = "PASS"
            d_icon = "✓"
            d_badge = "🟢 Favorable"
        elif distal_root_mm >= 0.8:
            d_status = "CAUTION"
            d_icon = "⚠"
            d_badge = "🟡 Proximity Warning"
            has_caution = True
        else:
            d_status = "DANGER"
            d_icon = "✕"
            d_badge = "🔴 Root Collision Risk"
            has_critical = True
        checklist.append({
            "item": f"Distal Root (FDI {distal_fdi})",
            "measured_mm": distal_root_mm,
            "threshold_mm": root_threshold_mm,
            "status": d_status,
            "icon": d_icon,
            "badge": d_badge,
            "note": f"{distal_root_mm:.1f} mm to adjacent root (recommended ≥ {root_threshold_mm:.1f} mm)",
        })

    # D. Buccal & Lingual plate checks
    if buccal_plate_mm >= cortical_threshold_mm:
        b_status, b_icon, b_badge = "PASS", "✓", "🟢 Favorable"
    else:
        b_status, b_icon, b_badge = "CAUTION", "⚠", "🟡 Thin Buccal Plate"
        has_caution = True
    checklist.append({
        "item": "Buccal Cortical Plate",
        "measured_mm": buccal_plate_mm,
        "threshold_mm": cortical_threshold_mm,
        "status": b_status,
        "icon": b_icon,
        "badge": b_badge,
        "note": f"{buccal_plate_mm:.1f} mm bone margin (recommended ≥ {cortical_threshold_mm:.1f} mm)",
    })

    if lingual_plate_mm >= cortical_threshold_mm:
        l_status, l_icon, l_badge = "PASS", "✓", "🟢 Favorable"
    else:
        l_status, l_icon, l_badge = "CAUTION", "⚠", "🟡 Thin Lingual Plate"
        has_caution = True
    checklist.append({
        "item": "Lingual Cortical Plate",
        "measured_mm": lingual_plate_mm,
        "threshold_mm": cortical_threshold_mm,
        "status": l_status,
        "icon": l_icon,
        "badge": l_badge,
        "note": f"{lingual_plate_mm:.1f} mm bone margin (recommended ≥ {cortical_threshold_mm:.1f} mm)",
    })

    # Direct collision check
    if collision_info["has_collision"]:
        has_critical = True

    # Compute overall status category
    if has_critical:
        overall_status = "HIGH RISK / REPOSITION"
        overall_badge = "🔴 HIGH RISK / REPOSITION"
        overall_color = "#DC2626"
    elif has_caution:
        overall_status = "REVIEW / LIMITED CLEARANCE"
        overall_badge = "🟡 REVIEW / LIMITED CLEARANCE"
        overall_color = "#D97706"
    else:
        overall_status = "PRELIMINARY FEASIBLE"
        overall_badge = "🟢 PRELIMINARY FEASIBLE"
        overall_color = "#16A34A"

    return {
        "canal_distance_mm": canal_dist_mm,
        "mesial_root_clearance_mm": mesial_root_mm,
        "distal_root_clearance_mm": distal_root_mm,
        "mesial_adjacent_fdi": mesial_fdi,
        "distal_adjacent_fdi": distal_fdi,
        "buccal_plate_clearance_mm": buccal_plate_mm,
        "lingual_plate_clearance_mm": lingual_plate_mm,
        "is_contained": is_contained,
        "planning_threshold_mm": planning_threshold_mm,
        "root_threshold_mm": root_threshold_mm,
        "cortical_threshold_mm": cortical_threshold_mm,
        "collision_info": collision_info,
        "nearest_canal_vox": nearest_canal_vox,
        "is_mandibular": is_mandibular,
        "checklist": checklist,
        "overall_status": overall_status,
        "overall_badge": overall_badge,
        "overall_color": overall_color,
    }
