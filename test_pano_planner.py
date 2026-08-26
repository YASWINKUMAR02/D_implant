"""
Comprehensive test suite for 2D Panoramic Tooth Instance Detection,
FDI Numbering, 32-Tooth Present/Missing Analysis & Implant Planning.
"""
import numpy as np
import cv2
from utils.panoramic_planner import (
    run_full_planning,
    render_planning_figure,
    render_debug_figure,
    render_site_detail_figure,
    generate_site_report,
    plans_to_dict_list,
    ALL_32_FDI_TEETH,
)

def test_full_pipeline():
    h, w = 512, 1024
    dummy_img = np.full((h, w, 3), 45, dtype=np.uint8)
    dummy_mask = np.zeros((h, w), dtype=np.uint8)

    # 1. Simulate Upper Maxillary Teeth (continuous row of teeth)
    for x in range(220, 800, 36):
        cv2.ellipse(dummy_mask, (x, 190), (16, 40), 0, 0, 360, 255, -1)

    # 2. Simulate Lower Mandibular Teeth with an EDENTULOUS GAP (missing teeth 44, 45, 46)
    # Molars: 48, 47 (x = 220, 260)
    cv2.ellipse(dummy_mask, (220, 340), (16, 35), 0, 0, 360, 255, -1)
    cv2.ellipse(dummy_mask, (260, 340), (16, 35), 0, 0, 360, 255, -1)
    # GAP: x = 280 to 450 (Missing 46, 45, 44)
    # Anterior & Left teeth: 43 to 38 (x = 460 to 800)
    for x in range(460, 800, 34):
        cv2.ellipse(dummy_mask, (x, 340), (15, 35), 0, 0, 360, 255, -1)

    print("Running redesigned 2D planning pipeline...")
    result = run_full_planning(
        dummy_img,
        dummy_mask,
        opg_width_mm=150.0,
        is_calibrated=False,
        min_gap_mm=4.5,
    )

    print(f"Detected teeth instances: {len(result.detected_teeth)}")
    print(f"Edentulous spaces: {len(result.edentulous_spaces)}")
    print(f"Total 32-tooth entries: {len(result.tooth_status_table)}")
    print(f"Implant candidate plans: {len(result.implant_plans)}")

    # Verify individual bounding box detections
    assert len(result.detected_teeth) >= 15, "Should detect individual teeth instances"
    for t in result.detected_teeth:
        assert len(t.bbox) == 4, "Every tooth must have (x1, y1, x2, y2) bounding box"
        assert t.width_px > 0 and t.height_px > 0, "Valid dimensions"
        assert 11 <= t.fdi <= 48, f"Valid FDI number: {t.fdi}"
        assert t.confidence >= 0.85, "High detection confidence"

    # Verify 32-tooth status table
    assert len(result.tooth_status_table) == 32, "Must contain exactly 32 permanent teeth"
    status_by_fdi = {e.fdi: e for e in result.tooth_status_table}
    for fdi in ALL_32_FDI_TEETH:
        assert fdi in status_by_fdi, f"FDI {fdi} must be present in table"

    # Verify that edentulous space was detected in lower right quadrant
    assert len(result.edentulous_spaces) >= 1, "Must detect the edentulous gap in lower arch"
    gap = result.edentulous_spaces[0]
    print(f"Detected Space: FDI {gap.suspected_fdis} (~{gap.width_mm:.1f} mm, region: {gap.arch_region})")
    assert gap.width_mm >= 5.0, "Gap width must be >= 5mm"

    # Test Renderers
    print("Testing render_planning_figure()...")
    fig_main = render_planning_figure(result, dummy_img)
    assert fig_main is not None

    print("Testing render_debug_figure()...")
    fig_debug = render_debug_figure(result, dummy_img)
    assert fig_debug is not None

    print("Testing render_site_detail_figure()...")
    fig_detail = render_site_detail_figure(result, dummy_img, result.implant_plans[0].site_fdi)
    assert fig_detail is not None

    print("Testing generate_site_report()...")
    rep = generate_site_report(result.implant_plans[0], result)
    assert "PRELIMINARY" in rep
    assert "CBCT" in rep

    print("Testing JSON serialization...")
    json_list = plans_to_dict_list(result.implant_plans)
    assert len(json_list) == len(result.implant_plans)

    print("\n[SUCCESS] Redesigned 2D Panoramic Tooth Detection & Planning tests PASSED!")

if __name__ == "__main__":
    test_full_pipeline()
