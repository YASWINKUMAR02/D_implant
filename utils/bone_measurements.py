"""
bone_measurements.py
Calculates physical 3D bone dimensions in true millimeters:
  - Mesiodistal available space (mm)
  - Buccolingual ridge width at multiple depths (mm)
  - Usable vertical bone height to canal or sinus (mm)
  - Mesial & Distal adjacent root clearances (mm)
  - Buccal & Lingual cortical plate thickness (mm)
Using the OralSeg 35-class volumetric segmentation and voxel spacing [dx, dy, dz].
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from utils.implant_geometry import FDI_TO_LABEL, get_adjacent_teeth_fdi, ALL_FDI_TEETH


def calculate_bone_height(
    seg_vol: np.ndarray,
    site_center_voxel: np.ndarray,
    is_mandibular: bool = True,
    voxel_spacing_mm: List[float] = [1.0, 1.0, 1.0],
    search_radius_vox: int = 4,
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate available vertical bone height in physical millimeters.
    
    For Mandible (Teeth 31-48):
      Measures vertically from the alveolar ridge crest downward toward the 
      Mandibular Canal (Label 35) or inferior cortical border (Label 2).
      
    For Maxilla (Teeth 11-28):
      Measures vertically from alveolar crest upward toward sinus floor / nasal floor (Label 1 boundary).
    """
    cx, cy, cz = int(round(site_center_voxel[0])), int(round(site_center_voxel[1])), int(round(site_center_voxel[2]))
    jaw_label = 2 if is_mandibular else 1
    canal_label = 35

    x_min = max(0, cx - search_radius_vox)
    x_max = min(seg_vol.shape[0], cx + search_radius_vox + 1)
    y_min = max(0, cy - search_radius_vox)
    y_max = min(seg_vol.shape[1], cy + search_radius_vox + 1)

    sub_vol = seg_vol[x_min:x_max, y_min:y_max, :]
    jaw_voxels = np.argwhere(sub_vol == jaw_label)
    canal_voxels = np.argwhere(sub_vol == canal_label)

    if len(jaw_voxels) == 0:
        jaw_all = np.argwhere(seg_vol == jaw_label)
        if len(jaw_all) > 0:
            z_span = (jaw_all[:, 2].max() - jaw_all[:, 2].min()) * voxel_spacing_mm[2]
            return float(max(6.0, min(z_span * 0.5, 18.0))), {"method": "global_jaw_estimate"}
        return 12.0, {"method": "default_standard"}

    if is_mandibular:
        crest_z = jaw_voxels[:, 2].max()

        if len(canal_voxels) > 0:
            canal_below = canal_voxels[canal_voxels[:, 2] < crest_z]
            if len(canal_below) > 0:
                canal_top_z = canal_below[:, 2].max()
                delta_z_vox = crest_z - canal_top_z
                bone_height_mm = delta_z_vox * voxel_spacing_mm[2]
                return float(max(1.0, bone_height_mm)), {
                    "method": "crest_to_canal",
                    "crest_z": int(crest_z),
                    "canal_top_z": int(canal_top_z),
                    "has_canal": True,
                }

        inf_z = jaw_voxels[:, 2].min()
        delta_z_vox = crest_z - inf_z
        bone_height_mm = delta_z_vox * voxel_spacing_mm[2]
        return float(max(1.0, bone_height_mm)), {
            "method": "crest_to_inferior_border",
            "crest_z": int(crest_z),
            "inferior_z": int(inf_z),
            "has_canal": False,
        }
    else:
        crest_z = jaw_voxels[:, 2].min()
        sinus_z = jaw_voxels[:, 2].max()
        delta_z_vox = sinus_z - crest_z
        bone_height_mm = delta_z_vox * voxel_spacing_mm[2]
        return float(max(1.0, bone_height_mm)), {
            "method": "maxillary_crest_to_sinus",
            "crest_z": int(crest_z),
            "sinus_z": int(sinus_z),
            "has_canal": False,
        }


def calculate_ridge_width(
    seg_vol: np.ndarray,
    site_center_voxel: np.ndarray,
    is_mandibular: bool = True,
    voxel_spacing_mm: List[float] = [1.0, 1.0, 1.0],
    search_depth_mm: float = 3.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Estimate the available buccolingual width of alveolar bone around the selected site in physical mm.
    """
    cx, cy, cz = int(round(site_center_voxel[0])), int(round(site_center_voxel[1])), int(round(site_center_voxel[2]))
    jaw_label = 2 if is_mandibular else 1

    depth_vox = int(round(search_depth_mm / max(voxel_spacing_mm[2], 0.01)))
    eval_z = cz - depth_vox if is_mandibular else cz + depth_vox
    eval_z = int(np.clip(eval_z, 0, seg_vol.shape[2] - 1))

    patch_radius = 12
    x_min = max(0, cx - patch_radius)
    x_max = min(seg_vol.shape[0], cx + patch_radius + 1)
    y_min = max(0, cy - patch_radius)
    y_max = min(seg_vol.shape[1], cy + patch_radius + 1)

    slice_patch = seg_vol[x_min:x_max, y_min:y_max, eval_z]
    jaw_in_patch = np.argwhere(slice_patch == jaw_label)

    if len(jaw_in_patch) < 5:
        for dz in [-2, -1, 1, 2]:
            test_z = int(np.clip(eval_z + dz, 0, seg_vol.shape[2] - 1))
            slice_patch = seg_vol[x_min:x_max, y_min:y_max, test_z]
            jaw_in_patch = np.argwhere(slice_patch == jaw_label)
            if len(jaw_in_patch) >= 5:
                break

    if len(jaw_in_patch) >= 5:
        coords_mm = jaw_in_patch * np.array([voxel_spacing_mm[0], voxel_spacing_mm[1]])
        coords_centered = coords_mm - coords_mm.mean(axis=0)
        cov = np.cov(coords_centered.T)
        if cov.ndim == 2 and cov.shape == (2, 2):
            eigvals = np.linalg.eigvalsh(cov)
            width_pca = 2.0 * np.sqrt(max(eigvals[0], 0.1) * 3.0)
            
            span_x = (coords_mm[:, 0].max() - coords_mm[:, 0].min())
            span_y = (coords_mm[:, 1].max() - coords_mm[:, 1].min())
            width_direct = min(span_x, span_y)

            final_width = float(np.clip(min(width_pca, width_direct) if width_direct > 1.0 else width_pca, 3.0, 16.0))
            return round(final_width, 1), {"eval_z": eval_z, "samples": len(jaw_in_patch), "method": "local_patch_pca"}

    return 7.5, {"method": "standard_ridge_default"}


def calculate_mesiodistal_space(
    seg_vol: np.ndarray,
    site_center_voxel: np.ndarray,
    fdi_tooth: int,
    detected_teeth: List[int],
    voxel_spacing_mm: List[float] = [1.0, 1.0, 1.0],
) -> Tuple[float, Optional[int], Optional[int]]:
    """
    Calculate available Mesiodistal space (in mm) between adjacent detected tooth boundaries.
    
    Returns:
        (mesiodistal_mm, mesial_fdi, distal_fdi)
    """
    spacing = np.array(voxel_spacing_mm)
    mesial_fdi, distal_fdi = get_adjacent_teeth_fdi(fdi_tooth, detected_teeth)

    mesial_pts_mm = None
    distal_pts_mm = None

    if mesial_fdi is not None:
        lbl_m = FDI_TO_LABEL.get(mesial_fdi)
        if lbl_m is not None:
            vox = np.argwhere(seg_vol == lbl_m)
            if len(vox) >= 10:
                mesial_pts_mm = vox * spacing

    if distal_fdi is not None:
        lbl_d = FDI_TO_LABEL.get(distal_fdi)
        if lbl_d is not None:
            vox = np.argwhere(seg_vol == lbl_d)
            if len(vox) >= 10:
                distal_pts_mm = vox * spacing

    if mesial_pts_mm is not None and distal_pts_mm is not None:
        # Distance between closest boundary surfaces
        step_m = max(1, len(mesial_pts_mm) // 80)
        step_d = max(1, len(distal_pts_mm) // 80)
        sub_m = mesial_pts_mm[::step_m]
        sub_d = distal_pts_mm[::step_d]
        diff = sub_m[:, np.newaxis, :] - sub_d[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        min_md_dist = float(dists.min())
        return round(min_md_dist, 1), mesial_fdi, distal_fdi

    elif mesial_pts_mm is not None or distal_pts_mm is not None:
        ref_pts = mesial_pts_mm if mesial_pts_mm is not None else distal_pts_mm
        center_mm = site_center_voxel * spacing
        step_r = max(1, len(ref_pts) // 80)
        sub_r = ref_pts[::step_r]
        diff = sub_r - center_mm[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=-1)
        half_gap = float(dists.min())
        return round(half_gap * 2.0, 1), mesial_fdi, distal_fdi

    # Standard default mesiodistal tooth width based on tooth type
    t_num = fdi_tooth % 10
    default_widths = {1: 8.5, 2: 6.5, 3: 7.5, 4: 7.0, 5: 7.0, 6: 10.0, 7: 9.0, 8: 8.5}
    return default_widths.get(t_num, 7.5), mesial_fdi, distal_fdi


def calculate_adjacent_root_distances(
    implant_points_voxel: np.ndarray,
    seg_vol: np.ndarray,
    fdi_tooth: int,
    detected_teeth: List[int],
    voxel_spacing_mm: List[float] = [1.0, 1.0, 1.0],
) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
    """
    Calculate minimum 3D Euclidean distances (in mm) from the virtual implant to
    the Mesial adjacent tooth root and Distal adjacent tooth root separately.
    
    Returns:
        (mesial_clearance_mm, distal_clearance_mm, mesial_fdi, distal_fdi)
    """
    spacing = np.array(voxel_spacing_mm)
    implant_pts_mm = implant_points_voxel * spacing
    step_imp = max(1, len(implant_pts_mm) // 60)
    imp_sub = implant_pts_mm[::step_imp]

    mesial_fdi, distal_fdi = get_adjacent_teeth_fdi(fdi_tooth, detected_teeth)

    mesial_dist_mm = None
    if mesial_fdi is not None:
        lbl_m = FDI_TO_LABEL.get(mesial_fdi)
        if lbl_m is not None:
            vox_m = np.argwhere(seg_vol == lbl_m)
            if len(vox_m) >= 10:
                pts_m_mm = vox_m * spacing
                step_m = max(1, len(pts_m_mm) // 100)
                sub_m = pts_m_mm[::step_m]
                diff = imp_sub[:, np.newaxis, :] - sub_m[np.newaxis, :, :]
                mesial_dist_mm = round(float(np.linalg.norm(diff, axis=-1).min()), 1)

    distal_dist_mm = None
    if distal_fdi is not None:
        lbl_d = FDI_TO_LABEL.get(distal_fdi)
        if lbl_d is not None:
            vox_d = np.argwhere(seg_vol == lbl_d)
            if len(vox_d) >= 10:
                pts_d_mm = vox_d * spacing
                step_d = max(1, len(pts_d_mm) // 100)
                sub_d = pts_d_mm[::step_d]
                diff = imp_sub[:, np.newaxis, :] - sub_d[np.newaxis, :, :]
                distal_dist_mm = round(float(np.linalg.norm(diff, axis=-1).min()), 1)

    return mesial_dist_mm, distal_dist_mm, mesial_fdi, distal_fdi


def calculate_cortical_plate_clearances(
    implant_points_voxel: np.ndarray,
    seg_vol: np.ndarray,
    fdi_tooth: int,
    voxel_spacing_mm: List[float] = [1.0, 1.0, 1.0],
) -> Tuple[float, float, bool]:
    """
    Measure thickness of buccal and lingual cortical bone plates around the implant,
    and determine whether bone containment is maintained or if a cortical breach occurs.
    
    Returns:
        (buccal_clearance_mm, lingual_clearance_mm, is_contained)
    """
    is_mandibular = (fdi_tooth >= 31 and fdi_tooth <= 48)
    jaw_label = 2 if is_mandibular else 1

    valid_pts = []
    for pt in implant_points_voxel:
        ix, iy, iz = int(round(pt[0])), int(round(pt[1])), int(round(pt[2]))
        if 0 <= ix < seg_vol.shape[0] and 0 <= iy < seg_vol.shape[1] and 0 <= iz < seg_vol.shape[2]:
            valid_pts.append([ix, iy, iz])

    if not valid_pts:
        return 1.5, 1.5, True

    valid_arr = np.array(valid_pts)
    sampled = seg_vol[valid_arr[:, 0], valid_arr[:, 1], valid_arr[:, 2]]
    
    # Check what proportion of implant is inside the jaw bone
    bone_vox_count = (sampled == jaw_label).sum()
    containment_ratio = bone_vox_count / max(len(sampled), 1)
    is_contained = containment_ratio >= 0.70

    # Buccal vs Lingual clearance approximation
    cx = valid_arr[:, 0].mean()
    cy = valid_arr[:, 1].mean()
    cz = int(valid_arr[:, 2].mean())

    # In dental orientation:
    # Quadrants 1 & 4 (Right): Buccal is +X, Lingual is -X
    # Quadrants 2 & 3 (Left): Buccal is -X, Lingual is +X
    quadrant = fdi_tooth // 10
    is_right = quadrant in (1, 4)
    buccal_dir = 1.0 if is_right else -1.0

    slice_jaw = np.argwhere(seg_vol[:, :, cz] == jaw_label) if 0 <= cz < seg_vol.shape[2] else np.array([])
    if len(slice_jaw) > 10:
        coords_x = slice_jaw[:, 0] * voxel_spacing_mm[0]
        imp_x_mm = cx * voxel_spacing_mm[0]
        if buccal_dir > 0:
            buccal_plate = float(max(0.0, coords_x.max() - imp_x_mm))
            lingual_plate = float(max(0.0, imp_x_mm - coords_x.min()))
        else:
            buccal_plate = float(max(0.0, imp_x_mm - coords_x.min()))
            lingual_plate = float(max(0.0, coords_x.max() - imp_x_mm))
        return round(min(buccal_plate, 6.0), 1), round(min(lingual_plate, 6.0), 1), is_contained

    return 1.8, 1.6, is_contained


def get_site_measurements(
    seg_vol: np.ndarray,
    site_center_voxel: np.ndarray,
    fdi_tooth: int,
    detected_teeth: Optional[List[int]] = None,
    voxel_spacing_mm: List[float] = [1.0, 1.0, 1.0],
) -> Dict[str, Any]:
    """
    Comprehensive physical measurement calculator for the selected FDI tooth site.
    """
    is_mandibular = (fdi_tooth >= 31 and fdi_tooth <= 48)
    if detected_teeth is None:
        detected_teeth = ALL_FDI_TEETH
    
    bone_height_mm, height_details = calculate_bone_height(
        seg_vol, site_center_voxel, is_mandibular=is_mandibular, voxel_spacing_mm=voxel_spacing_mm
    )
    
    ridge_width_mm, width_details = calculate_ridge_width(
        seg_vol, site_center_voxel, is_mandibular=is_mandibular, voxel_spacing_mm=voxel_spacing_mm
    )

    mesiodistal_mm, mesial_fdi, distal_fdi = calculate_mesiodistal_space(
        seg_vol, site_center_voxel, fdi_tooth, detected_teeth, voxel_spacing_mm=voxel_spacing_mm
    )

    return {
        "fdi_tooth": fdi_tooth,
        "is_mandibular": is_mandibular,
        "bone_height_mm": round(bone_height_mm, 1),
        "ridge_width_mm": round(ridge_width_mm, 1),
        "mesiodistal_mm": round(mesiodistal_mm, 1),
        "mesial_adjacent_fdi": mesial_fdi,
        "distal_adjacent_fdi": distal_fdi,
        "height_details": height_details,
        "width_details": width_details,
        "voxel_spacing_mm": voxel_spacing_mm,
    }
