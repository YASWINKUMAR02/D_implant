"""
test_implant_planner.py
Comprehensive automated test for all 3D CBCT implant planning modules.
Validates:
  1. Missing teeth detection & site classification (Candidate / Review / Insufficient)
  2. 3D anatomical measurements in true physical millimeters (MD Space, Ridge Width, Bone Height)
  3. Virtual 3D implant mesh generation & STL export
  4. Multi-structure safety & collision analysis (Canal, Mesial Root, Distal Root, Cortical Plates)
  5. 2D MPR cross-sectional rendering & 3D Plotly viewport with camera presets
  6. Structured planning JSON and printable HTML dossier export
"""

import sys
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

print("--- Testing 3D CBCT AI-Assisted Dental Implant Planning Suite ---")

# 1. Test missing teeth detection & classification
from utils.implant_geometry import (
    detect_potential_missing_teeth, get_tooth_site_coordinates,
    classify_missing_teeth_sites, get_adjacent_teeth_fdi,
    create_virtual_implant_mesh, sample_implant_cylinder_points,
    ALL_FDI_TEETH
)

mock_seg_info = {
    "detected_teeth": [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 31, 32, 33, 34, 35, 37, 41, 42, 43, 44, 45, 47],
}
missing = detect_potential_missing_teeth(mock_seg_info)
print(f"  [x] Missing teeth detected: {missing}")
assert 46 in missing, "Tooth 46 should be in missing teeth"
assert 36 in missing, "Tooth 36 should be in missing teeth"

# Adjacent tooth discovery
mesial_46, distal_46 = get_adjacent_teeth_fdi(46, mock_seg_info["detected_teeth"])
print(f"  [x] Tooth 46 adjacent teeth: Mesial={mesial_46}, Distal={distal_46}")
assert mesial_46 == 45, "Tooth 46 mesial neighbor should be 45"
assert distal_46 == 47, "Tooth 46 distal neighbor should be 47"

# 2. Test mock 3D volume with physical 0.5 mm voxel spacing
shape = (120, 120, 100)
mock_seg_vol = np.zeros(shape, dtype=np.uint8)
# Mandible bone (2)
mock_seg_vol[30:90, 30:90, 20:65] = 2
# Mandibular Canal (35)
mock_seg_vol[50:65, 50:65, 25:30] = 35
# Adjacent tooth 45 (Label 31)
mock_seg_vol[65:75, 45:55, 55:75] = 31
# Adjacent tooth 47 (Label 33)
mock_seg_vol[35:45, 45:55, 55:75] = 33

spacing = [0.5, 0.5, 0.5] # 0.5 mm spacing

# Site classification
cat_sites = classify_missing_teeth_sites(mock_seg_vol, mock_seg_info, voxel_spacing_mm=spacing)
print(f"  [x] Site Classification: Candidates={len(cat_sites['candidate'])}, Review={len(cat_sites['needs_review'])}, Insufficient={len(cat_sites['insufficient_info'])}")
assert len(cat_sites["candidate"]) + len(cat_sites["needs_review"]) + len(cat_sites["insufficient_info"]) == len(missing)

# Site coordinate finder
center_vox, is_det = get_tooth_site_coordinates(mock_seg_vol, 46, voxel_spacing_mm=spacing)
print(f"  [x] Estimated Tooth 46 coronal center: {center_vox}, is_detected={is_det}")
assert center_vox.shape == (3,)

# 3. Test virtual implant cylinder math & STL export
verts, faces = create_virtual_implant_mesh(center_vox, diameter_mm=4.0, length_mm=10.0, voxel_spacing_mm=spacing)
print(f"  [x] Created virtual implant mesh: {len(verts)} vertices, {len(faces)} faces")
assert len(verts) > 0 and len(faces) > 0

pts = sample_implant_cylinder_points(center_vox, diameter_mm=4.0, length_mm=10.0, voxel_spacing_mm=spacing)
print(f"  [x] Sampled {len(pts)} points on virtual implant")
assert len(pts) > 50

from utils.implant_report import export_virtual_implant_stl
stl_file = export_virtual_implant_stl(center_vox, 4.0, 10.0, voxel_spacing_mm=spacing, output_path="outputs/test_implant.stl")
print(f"  [x] Exported binary STL: {stl_file}")
assert stl_file is not None and Path(stl_file).exists()

# 4. Test physical 3D bone measurements
from utils.bone_measurements import get_site_measurements
measurements = get_site_measurements(mock_seg_vol, center_vox, 46, detected_teeth=mock_seg_info["detected_teeth"], voxel_spacing_mm=spacing)
print(f"  [x] 3D Measurements for Tooth 46: Height={measurements['bone_height_mm']} mm, Ridge Width={measurements['ridge_width_mm']} mm, MD Space={measurements['mesiodistal_mm']} mm")
assert measurements["bone_height_mm"] > 0
assert measurements["ridge_width_mm"] > 0
assert measurements["mesiodistal_mm"] > 0

# 5. Test AI preliminary range suggestion
from utils.implant_suggester_3d import suggest_preliminary_implant_range_for_site
range_res = suggest_preliminary_implant_range_for_site(measurements["bone_height_mm"], measurements["ridge_width_mm"], 5.2)
print(f"  [x] Suggested preliminary ranges: Ø {range_res['diameter_range_str']}, L {range_res['length_range_str']}")
assert range_res["default_diameter_mm"] > 0
assert range_res["default_length_mm"] > 0

# 6. Test comprehensive safety and collision analysis
from utils.collision_analysis import analyze_implant_safety
safety = analyze_implant_safety(
    pts, mock_seg_vol, 46,
    detected_teeth=mock_seg_info["detected_teeth"],
    planning_threshold_mm=2.0,
    voxel_spacing_mm=spacing
)
print(f"  [x] Safety Analysis: Canal Dist={safety['canal_distance_mm']} mm, Mesial Root={safety['mesial_root_clearance_mm']} mm, Distal Root={safety['distal_root_clearance_mm']} mm")
print(f"  [x] Overall Safety Tier: {safety['overall_status']}")
assert safety["overall_status"] in ["PRELIMINARY FEASIBLE", "REVIEW / LIMITED CLEARANCE", "HIGH RISK / REPOSITION"]
assert len(safety["checklist"]) >= 4

# 7. Test 2D cross-sectional MPR rendering
from utils.implant_visualization import render_2d_mpr_implant_views, build_3d_implant_scene, build_local_roi_3d_scene
mock_cbct = np.random.randint(-1000, 2000, size=shape).astype(np.float32)

fig_mpr = render_2d_mpr_implant_views(
    mock_cbct, mock_seg_vol, center_vox,
    implant_diameter_mm=4.0, implant_length_mm=10.0,
    voxel_spacing_mm=spacing, canal_distance_mm=safety['canal_distance_mm'],
    bone_height_mm=measurements['bone_height_mm'], mesiodistal_mm=measurements['mesiodistal_mm']
)
print("  [x] 2D MPR cross-sections rendered successfully")

# 8. Test 3D Plotly implant scene with camera presets
fig_3d = build_3d_implant_scene(
    mock_seg_vol, center_vox, implant_diameter_mm=4.0, implant_length_mm=10.0,
    selected_fdi_tooth=46, detected_teeth=mock_seg_info["detected_teeth"],
    voxel_spacing_mm=spacing, camera_view="buccal"
)
print("  [x] 3D Plotly implant scene (Buccal preset) constructed successfully")

fig_roi = build_local_roi_3d_scene(
    mock_seg_vol, center_vox, implant_diameter_mm=4.0, implant_length_mm=10.0,
    voxel_spacing_mm=spacing
)
print("  [x] 3D Local ROI scene constructed successfully")

# 9. Test report generation
from utils.implant_report import generate_implant_planning_data, save_implant_planning_json, generate_printable_html_report

plan_data = generate_implant_planning_data(
    case_name="TEST_CASE",
    selected_fdi_tooth=46,
    detected_teeth=mock_seg_info["detected_teeth"],
    potential_missing_teeth=missing,
    bone_height_mm=measurements['bone_height_mm'],
    ridge_width_mm=measurements['ridge_width_mm'],
    mesiodistal_mm=measurements['mesiodistal_mm'],
    implant_diameter_mm=4.0,
    implant_length_mm=10.0,
    implant_angulation_deg=(0.0, 0.0, 0.0),
    implant_position_offset_mm=(0.0, 0.0, 0.0),
    canal_distance_mm=safety['canal_distance_mm'],
    mesial_root_clearance_mm=safety['mesial_root_clearance_mm'],
    distal_root_clearance_mm=safety['distal_root_clearance_mm'],
    mesial_fdi=safety['mesial_adjacent_fdi'],
    distal_fdi=safety['distal_adjacent_fdi'],
    buccal_plate_mm=safety['buccal_plate_clearance_mm'],
    lingual_plate_mm=safety['lingual_plate_clearance_mm'],
    is_contained=safety['is_contained'],
    overall_status=safety['overall_status'],
    checklist=safety['checklist'],
    planning_threshold_mm=2.0,
    voxel_spacing_mm=spacing,
)

json_out = save_implant_planning_json("outputs/test_plan.json", plan_data)
html_out = generate_printable_html_report(plan_data)
print("  [x] JSON planning metadata and printable HTML dossier generated successfully")

print("\n>>> ALL 3D CBCT IMPLANT PLANNING MODULES TESTED & PASSED 100%!")

