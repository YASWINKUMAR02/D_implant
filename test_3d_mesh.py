"""
test_3d_mesh.py
Verification script for improved 3D Dental CBCT mesh extraction and visualization.
"""

import sys
import numpy as np

def test_3d_pipeline():
    print("Testing 3D mesh extraction & visualization pipeline...")

    from utils.visualization import (
        clean_binary_mask,
        smooth_and_simplify_mesh,
        extract_mesh_for_label,
        build_3d_interactive_figure,
        export_binary_stl,
        TOOTH_LABEL_TO_FDI,
        FDI_3D_COLORS,
    )

    # 1. Create a synthetic test segmentation volume
    vol = np.zeros((80, 80, 80), dtype=np.uint8)
    
    # Maxilla (label 1)
    vol[10:35, 15:65, 45:65] = 1
    # Mandible (label 2)
    vol[45:75, 15:65, 15:35] = 2
    # Tooth 11 (label 3 -> FDI 11)
    vol[20:30, 35:45, 40:50] = 3
    # Tooth 21 (label 11 -> FDI 21)
    vol[20:30, 45:55, 40:50] = 11
    # Tooth 46 (label 32 -> FDI 46)
    vol[50:60, 20:30, 30:40] = 32
    # Mandibular Canal (label 35)
    vol[55:60, 20:60, 20:25] = 35
    # Small isolated noise voxels
    vol[2, 2, 2] = 3
    vol[78, 78, 78] = 2

    spacing = (0.4, 0.4, 0.4)

    # 2. Test noise cleaning
    noisy_mask = (vol == 3)
    clean_mask = clean_binary_mask(noisy_mask, min_voxels=30, keep_largest_n=1)
    assert clean_mask[2, 2, 2] == False, "Noise voxel was not filtered!"
    assert clean_mask[25, 40, 45] == True, "Main tooth body was incorrectly removed!"
    print("  [x] Connected component noise filtering passed.")

    # 3. Test mesh extraction with physical spacing and smoothing
    verts, faces = extract_mesh_for_label(
        vol, [3],
        voxel_spacing=spacing,
        smoothing_iterations=10,
    )
    assert verts is not None and len(verts) > 0, "Failed to extract mesh for Tooth 11"
    assert faces is not None and len(faces) > 0, "Failed to extract faces for Tooth 11"
    # Physical spacing check: max coordinate should roughly correspond to 80 * 0.4 = 32mm
    assert verts.max() < 35.0, f"Vertices not scaled properly: max={verts.max()}"
    print(f"  [x] Marching cubes with physical spacing & Taubin smoothing passed ({len(verts)} verts, {len(faces)} faces).")

    # 4. Test Plotly interactive figure generation with all features
    fig = build_3d_interactive_figure(
        vol,
        voxel_spacing=spacing,
        show_maxilla=True,
        show_mandible=True,
        show_teeth=True,
        show_canal=True,
        maxilla_opacity=0.28,
        mandible_opacity=0.28,
        teeth_opacity=1.0,
        canal_opacity=1.0,
        smoothing_iterations=10,
        camera_preset="anterior",
    )
    
    trace_names = [t.name for t in fig.data]
    print(f"  [x] Plotly figure generated {len(fig.data)} traces: {trace_names}")
    assert any("Maxilla" in name for name in trace_names), "Maxilla trace missing"
    assert any("Mandible" in name for name in trace_names), "Mandible trace missing"
    assert any("11" in name for name in trace_names), "Tooth 11 trace missing"
    assert any("Canal" in name for name in trace_names), "Canal trace missing"

    # 5. Test "Teeth Only" preset
    fig_teeth_only = build_3d_interactive_figure(
        vol,
        voxel_spacing=spacing,
        show_maxilla=False,
        show_mandible=False,
        show_teeth=True,
        show_canal=False,
    )
    teeth_traces = [t.name for t in fig_teeth_only.data]
    assert len(teeth_traces) == 3, f"Expected 3 teeth traces, got {len(teeth_traces)}"
    print("  [x] Teeth Only preset mode passed.")

    # 6. Test binary STL export
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        export_binary_stl(verts, faces, tmp_path)
        assert os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 84, "STL export failed"
        print(f"  [x] Binary STL export passed ({os.path.getsize(tmp_path):,} bytes).")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print("\nAll 3D mesh extraction & visualization tests passed successfully!")

if __name__ == "__main__":
    test_3d_pipeline()
