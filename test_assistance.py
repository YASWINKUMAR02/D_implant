"""
test_assistance.py
"""
import cv2
import numpy as np
from utils.panoramic_assistance import (
    run_tooth_segmentation_and_numbering,
    analyze_tooth_status_and_implant_sites,
    render_assistance_figure,
)

def test_assistance_module():
    h, w = 512, 1024
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    
    print("Testing Stage 1 & 2: Tooth Segmentation & FDI Numbering...")
    instances, curves = run_tooth_segmentation_and_numbering(img, opg_width_mm=150.0)
    print(f"Detected instances: {len(instances)}")
    
    print("Testing Stage 3 & 4: 32-Tooth Status & Implant Sites...")
    status_records, sites = analyze_tooth_status_and_implant_sites(instances, (h, w), px_per_mm=1024/150.0)
    print(f"Total 32-tooth records: {len(status_records)}")
    print(f"Potential implant sites: {len(sites)}")
    
    print("Testing Figure Rendering...")
    fig = render_assistance_figure(img, instances, sites, curves, px_per_mm=1024/150.0)
    assert fig is not None
    print("[SUCCESS] Assistance pipeline verified!")

if __name__ == "__main__":
    test_assistance_module()
