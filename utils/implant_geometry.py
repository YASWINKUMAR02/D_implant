"""
implant_geometry.py
Handles 3D virtual implant cylinder geometry, dental arch trajectories,
adjacent tooth tracking, site candidate classification, and physical coordinate transformations.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from utils.oralseg_inference import LABEL_MAP, TOOTH_FDI_MAP

# Inverted mapping: FDI number (e.g. 46) -> Label Index (e.g. 32)
FDI_TO_LABEL = {v: k for k, v in TOOTH_FDI_MAP.items()}
LABEL_TO_FDI = dict(TOOTH_FDI_MAP)

# All 32 adult permanent teeth in FDI notation
ALL_FDI_TEETH = [
    18, 17, 16, 15, 14, 13, 12, 11,
    21, 22, 23, 24, 25, 26, 27, 28,
    48, 47, 46, 45, 44, 43, 42, 41,
    31, 32, 33, 34, 35, 36, 37, 38,
]

# Quadrant ordering
QUADRANTS = {
    1: [18, 17, 16, 15, 14, 13, 12, 11],  # Maxillary Right (Distal to Mesial)
    2: [21, 22, 23, 24, 25, 26, 27, 28],  # Maxillary Left  (Mesial to Distal)
    4: [48, 47, 46, 45, 44, 43, 42, 41],  # Mandibular Right (Distal to Mesial)
    3: [31, 32, 33, 34, 35, 36, 37, 38],  # Mandibular Left  (Mesial to Distal)
}


def detect_potential_missing_teeth(seg_info: Dict[str, Any]) -> List[int]:
    """
    Compare expected 32 FDI teeth against OralSeg detected teeth.
    Returns list of FDI numbers for teeth marked as 'Not Detected'.
    """
    detected_set = set(seg_info.get("detected_teeth", []))
    missing = [tooth for tooth in ALL_FDI_TEETH if tooth not in detected_set]
    return missing


def get_adjacent_teeth_fdi(fdi_tooth: int, detected_teeth: List[int]) -> Tuple[Optional[int], Optional[int]]:
    """
    Determine the Mesial and Distal adjacent teeth for an FDI site from detected teeth.
    
    Returns:
        (mesial_fdi, distal_fdi)
    """
    q = fdi_tooth // 10
    t_num = fdi_tooth % 10
    det_set = set(detected_teeth)

    # In FDI notation:
    # Mesial = closer to midline (t_num - 1, or crossing midline if t_num == 1)
    # Distal = further from midline (t_num + 1)
    
    # 1. Mesial neighbor
    mesial_fdi = None
    if t_num > 1:
        cand = q * 10 + (t_num - 1)
        if cand in det_set:
            mesial_fdi = cand
    else:
        # Crossing midline (e.g. 11 mesial is 21, 41 mesial is 31)
        midline_opp = {11: 21, 21: 11, 41: 31, 31: 41}
        cand = midline_opp.get(fdi_tooth)
        if cand and cand in det_set:
            mesial_fdi = cand

    # 2. Distal neighbor
    distal_fdi = None
    if t_num < 8:
        cand = q * 10 + (t_num + 1)
        if cand in det_set:
            distal_fdi = cand

    return mesial_fdi, distal_fdi


def classify_missing_teeth_sites(
    seg_vol: np.ndarray,
    seg_info: Dict[str, Any],
    voxel_spacing_mm: List[float] = [1.0, 1.0, 1.0],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Categorize all missing teeth into three distinct clinical candidate tiers:
      - 'candidate': Clear alveolar bone ridge present + adjacent tooth references.
      - 'needs_review': Jaw bone detected but thin ridge or bounded by single tooth.
      - 'insufficient_info': Jaw region poorly segmented or unidentifiable.
    """
    missing = detect_potential_missing_teeth(seg_info)
    detected = seg_info.get("detected_teeth", [])
    det_set = set(detected)

    categorized = {
        "candidate": [],
        "needs_review": [],
        "insufficient_info": [],
    }

    for fdi in missing:
        is_mandibular = (fdi >= 31 and fdi <= 48)
        jaw_label = 2 if is_mandibular else 1
        mesial_fdi, distal_fdi = get_adjacent_teeth_fdi(fdi, detected)
        
        center_vox, is_det = get_tooth_site_coordinates(seg_vol, fdi, voxel_spacing_mm=voxel_spacing_mm)
        cx, cy, cz = int(np.clip(round(center_vox[0]), 0, seg_vol.shape[0]-1)), \
                     int(np.clip(round(center_vox[1]), 0, seg_vol.shape[1]-1)), \
                     int(np.clip(round(center_vox[2]), 0, seg_vol.shape[2]-1))

        # Check local bone density
        sub = seg_vol[max(0, cx-10):min(seg_vol.shape[0], cx+11),
                      max(0, cy-10):min(seg_vol.shape[1], cy+11),
                      max(0, cz-15):min(seg_vol.shape[2], cz+16)]
        jaw_voxels = (sub == jaw_label).sum()

        has_neighbors = (mesial_fdi is not None) or (distal_fdi is not None)
        has_both_neighbors = (mesial_fdi is not None) and (distal_fdi is not None)

        site_entry = {
            "fdi": fdi,
            "mesial_fdi": mesial_fdi,
            "distal_fdi": distal_fdi,
            "center_vox": center_vox,
            "is_mandibular": is_mandibular,
        }

        if jaw_voxels >= 300 and has_neighbors:
            if has_both_neighbors or jaw_voxels >= 800:
                site_entry["category"] = "candidate"
                site_entry["badge"] = "🟢 Candidate Site"
                site_entry["note"] = "Alveolar ridge and adjacent boundary references detected."
                categorized["candidate"].append(site_entry)
            else:
                site_entry["category"] = "needs_review"
                site_entry["badge"] = "🟡 Needs Review"
                site_entry["note"] = "Single adjacent tooth detected — manual positioning recommended."
                categorized["needs_review"].append(site_entry)
        elif jaw_voxels >= 100:
            site_entry["category"] = "needs_review"
            site_entry["badge"] = "🟡 Needs Review"
            site_entry["note"] = "Limited local bone volume detected in ROI."
            categorized["needs_review"].append(site_entry)
        else:
            site_entry["category"] = "insufficient_info"
            site_entry["badge"] = "⚪ Insufficient Data"
            site_entry["note"] = "Alveolar bone segmentation insufficient at this quadrant coordinate."
            categorized["insufficient_info"].append(site_entry)

    return categorized


def get_tooth_site_coordinates(
    seg_vol: np.ndarray,
    fdi_tooth: int,
    voxel_spacing_mm: List[float] = [1.0, 1.0, 1.0],
) -> Tuple[np.ndarray, bool]:
    """
    Find or estimate 3D voxel center (x, y, z) for the specified FDI tooth.
    
    Returns:
        (center_voxels, is_detected)
    """
    label_idx = FDI_TO_LABEL.get(fdi_tooth)
    if label_idx is not None:
        coords = np.argwhere(seg_vol == label_idx)
        if len(coords) >= 30:
            # Tooth is present - use its coronal crest centroid
            is_mandibular = (fdi_tooth >= 31 and fdi_tooth <= 48)
            # For coronal positioning: take top 25% for mandible, bottom 25% for maxilla
            cz = np.percentile(coords[:, 2], 75 if is_mandibular else 25)
            centroid = np.array([coords[:, 0].mean(), coords[:, 1].mean(), cz])
            return centroid, True

    is_mandibular = (fdi_tooth >= 31 and fdi_tooth <= 48)
    jaw_label = 2 if is_mandibular else 1

    quadrant = fdi_tooth // 10
    quadrant_teeth = QUADRANTS.get(quadrant, [])
    
    found_centroids = []
    found_indices = []
    for idx, t in enumerate(quadrant_teeth):
        lbl = FDI_TO_LABEL.get(t)
        if lbl is not None:
            c = np.argwhere(seg_vol == lbl)
            if len(c) >= 30:
                cz = np.percentile(c[:, 2], 75 if is_mandibular else 25)
                found_centroids.append(np.array([c[:, 0].mean(), c[:, 1].mean(), cz]))
                found_indices.append(idx)

    target_idx = quadrant_teeth.index(fdi_tooth) if fdi_tooth in quadrant_teeth else 0

    if len(found_centroids) >= 2:
        # Interpolate / extrapolate along quadrant curve
        xs = np.interp(target_idx, found_indices, [pt[0] for pt in found_centroids])
        ys = np.interp(target_idx, found_indices, [pt[1] for pt in found_centroids])
        zs = np.interp(target_idx, found_indices, [pt[2] for pt in found_centroids])
        
        # Snap z to the local alveolar crest of the jaw bone
        ix, iy = int(np.clip(round(xs), 0, seg_vol.shape[0]-1)), int(np.clip(round(ys), 0, seg_vol.shape[1]-1))
        local_jaw = np.argwhere(seg_vol[max(0, ix-4):min(seg_vol.shape[0], ix+5),
                                        max(0, iy-4):min(seg_vol.shape[1], iy+5), :] == jaw_label)
        if len(local_jaw) > 0:
            zs = float(local_jaw[:, 2].max() if is_mandibular else local_jaw[:, 2].min())

        return np.array([xs, ys, zs]), False
        
    elif len(found_centroids) == 1:
        base = found_centroids[0]
        offset = (target_idx - found_indices[0]) * 8.5 / max(voxel_spacing_mm[0], 0.1)
        # Quadrants 1 & 4 are Right side (offset -X or +Y depending on arch)
        is_right = quadrant in (1, 4)
        x_dir = -1.0 if is_right else 1.0
        return np.array([base[0] + x_dir * offset, base[1] + offset * 0.4, base[2]]), False

    # Fallback to jaw crest
    jaw_coords = np.argwhere(seg_vol == jaw_label)
    if len(jaw_coords) > 0:
        jaw_center = jaw_coords.mean(axis=0)
        t_num = fdi_tooth % 10
        is_right = quadrant in (1, 4)
        x_dir = -1.0 if is_right else 1.0
        
        offset_x = x_dir * (12.0 + t_num * 4.5) / max(voxel_spacing_mm[0], 0.1)
        offset_y = (t_num * 3.5 - 12.0) / max(voxel_spacing_mm[1], 0.1)
        z_pos = float(np.percentile(jaw_coords[:, 2], 90 if is_mandibular else 10))
        
        est_point = np.array([
            np.clip(jaw_center[0] + offset_x, 0, seg_vol.shape[0] - 1),
            np.clip(jaw_center[1] + offset_y, 0, seg_vol.shape[1] - 1),
            z_pos
        ])
        return est_point, False

    return np.array([seg_vol.shape[0] / 2.0, seg_vol.shape[1] / 2.0, seg_vol.shape[2] / 2.0]), False


def create_virtual_implant_mesh(
    center_voxel: np.ndarray,
    diameter_mm: float,
    length_mm: float,
    angulation_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    voxel_spacing_mm: List[float] = [1.0, 1.0, 1.0],
    num_segments: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 3D vertices and triangular faces of a virtual implant cylinder.
    Includes coronal prosthetic collar and apical taper.
    
    Args:
        center_voxel: 3D point (x, y, z) in voxel space where the coronal crest sits.
        diameter_mm: Implant diameter in mm.
        length_mm: Implant length in mm.
        angulation_deg: (rx_buccolingual, ry_mesiodistal, rz_axial) in degrees.
        voxel_spacing_mm: [dx, dy, dz].
        num_segments: Radial circle resolution.
        
    Returns:
        verts_voxel: (N, 3) vertices in voxel space.
        faces: (M, 3) triangular face indices.
    """
    radius_vox_x = (diameter_mm / 2.0) / max(voxel_spacing_mm[0], 0.01)
    radius_vox_y = (diameter_mm / 2.0) / max(voxel_spacing_mm[1], 0.01)
    length_vox_z = length_mm / max(voxel_spacing_mm[2], 0.01)

    # Multi-tier vertical rings (Coronal collar -> Body -> Apical taper)
    z_levels = [
        (0.0, 1.0),                  # Coronal collar top
        (-0.15 * length_vox_z, 1.0), # Collar bottom
        (-0.75 * length_vox_z, 0.95),# Mid-body taper start
        (-0.95 * length_vox_z, 0.65),# Apical taper
        (-length_vox_z, 0.35),       # Apical tip
    ]

    theta = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    ring_verts = []
    for z_pos, r_scale in z_levels:
        rx = radius_vox_x * r_scale
        ry = radius_vox_y * r_scale
        ring = np.column_stack([rx * cos_t, ry * sin_t, np.full(num_segments, z_pos)])
        ring_verts.append(ring)

    # Top center and bottom center vertices
    top_center = np.array([[0.0, 0.0, 0.0]])
    bottom_center = np.array([[0.0, 0.0, -length_vox_z]])

    all_verts = np.vstack(ring_verts + [top_center, bottom_center])

    # Apply 3D Rotation
    rx = np.radians(angulation_deg[0])
    ry = np.radians(angulation_deg[1])
    rz = np.radians(angulation_deg[2])

    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx

    rotated_verts = (R @ all_verts.T).T + center_voxel

    # Triangulate
    faces = []
    top_center_idx = len(all_verts) - 2
    bottom_center_idx = len(all_verts) - 1

    # Top cap
    for i in range(num_segments):
        next_i = (i + 1) % num_segments
        faces.append([top_center_idx, i, next_i])

    # Side bands between adjacent rings
    for ring_idx in range(len(z_levels) - 1):
        r1_start = ring_idx * num_segments
        r2_start = (ring_idx + 1) * num_segments
        for i in range(num_segments):
            next_i = (i + 1) % num_segments
            v1 = r1_start + i
            v2 = r1_start + next_i
            v3 = r2_start + next_i
            v4 = r2_start + i
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])

    # Bottom cap
    last_ring_start = (len(z_levels) - 1) * num_segments
    for i in range(num_segments):
        next_i = (i + 1) % num_segments
        faces.append([bottom_center_idx, last_ring_start + next_i, last_ring_start + i])

    return np.array(rotated_verts), np.array(faces)


def sample_implant_cylinder_points(
    center_voxel: np.ndarray,
    diameter_mm: float,
    length_mm: float,
    angulation_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    voxel_spacing_mm: List[float] = [1.0, 1.0, 1.0],
    num_samples: int = 250,
) -> np.ndarray:
    """
    Sample dense 3D points on the surface and core of the virtual implant cylinder.
    Used for collision and distance testing against mandibular canal, adjacent roots, and bone.
    """
    radius_vox_x = (diameter_mm / 2.0) / max(voxel_spacing_mm[0], 0.01)
    radius_vox_y = (diameter_mm / 2.0) / max(voxel_spacing_mm[1], 0.01)
    length_vox_z = length_mm / max(voxel_spacing_mm[2], 0.01)

    z_samples = np.linspace(0, -length_vox_z, 25)
    r_fractions = [0.0, 0.4, 0.75, 1.0]
    theta_samples = np.linspace(0, 2 * np.pi, 16, endpoint=False)

    pts = []
    for z in z_samples:
        taper = 1.0 if z > -0.75 * length_vox_z else (1.0 - 0.4 * ((-z / length_vox_z) - 0.75) / 0.25)
        for r_frac in r_fractions:
            if r_frac == 0.0:
                pts.append([0.0, 0.0, z])
            else:
                for th in theta_samples:
                    pts.append([
                        r_frac * radius_vox_x * taper * np.cos(th),
                        r_frac * radius_vox_y * taper * np.sin(th),
                        z
                    ])

    pts = np.array(pts)

    rx = np.radians(angulation_deg[0])
    ry = np.radians(angulation_deg[1])
    rz = np.radians(angulation_deg[2])

    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx

    world_pts = (R @ pts.T).T + center_voxel
    return world_pts

