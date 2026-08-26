"""
yolo_tooth_detector.py
======================
32-Class FDI Tooth Detection & Missing-Tooth Gap Analysis using YOLOv8.

Processes panoramic X-rays to:
1. Predict individual tooth bounding boxes and FDI numbers (11..48).
2. Perform anatomical sequence gap analysis to identify missing teeth.
3. Compute exact physical/pixel edentulous gap dimensions between adjacent teeth.
4. Generate candidate implant sites and full 32-tooth status entries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

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

FDI_NAMES = [
    '11', '12', '13', '14', '15', '16', '17', '18',
    '21', '22', '23', '24', '25', '26', '27', '28',
    '31', '32', '33', '34', '35', '36', '37', '38',
    '41', '42', '43', '44', '45', '46', '47', '48'
]

@dataclass
class YoloToothBox:
    fdi: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    cx: float
    cy: float
    width_px: float
    height_px: float
    confidence: float
    is_upper: bool
    arch_region: str


class YoloToothDetector:
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or YOLO_MODEL_PATH
        self.model = None
        self._load_model()

    def _load_model(self):
        if self.model_path.exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(self.model_path))
                logger.info(f"Loaded YOLO FDI Tooth model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load YOLO model: {e}")
                self.model = None

    def is_available(self) -> bool:
        return self.model is not None

    def predict(
        self,
        img_bgr: np.ndarray,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
    ) -> List[YoloToothBox]:
        if not self.is_available():
            return []

        results = self.model.predict(
            source=img_bgr,
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False,
        )

        boxes_list: List[YoloToothBox] = []
        if not results or len(results) == 0:
            return boxes_list

        res = results[0]
        boxes = res.boxes
        if boxes is None:
            return boxes_list

        h, w = img_bgr.shape[:2]

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            fdi_str = res.names.get(cls_id, str(cls_id))
            try:
                fdi = int(fdi_str)
            except ValueError:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            width_px = float(x2 - x1)
            height_px = float(y2 - y1)

            is_upper = (fdi < 30)
            q = fdi // 10
            quad_map = {1: "Maxillary Right (Q1)", 2: "Maxillary Left (Q2)", 3: "Mandibular Left (Q3)", 4: "Mandibular Right (Q4)"}
            arch_region = quad_map.get(q, "Unknown")

            boxes_list.append(YoloToothBox(
                fdi=fdi,
                bbox=(x1, y1, x2, y2),
                cx=cx, cy=cy,
                width_px=width_px,
                height_px=height_px,
                confidence=round(conf, 2),
                is_upper=is_upper,
                arch_region=arch_region,
            ))

        # Sort by FDI
        boxes_list.sort(key=lambda t: t.fdi)
        return boxes_list


def analyze_missing_teeth_from_yolo(
    detected_teeth: List[YoloToothBox],
    img_shape: Tuple[int, int],
    px_per_mm: float,
) -> Tuple[List[dict], List[dict]]:
    """
    Analyzes sequence gaps and spatial coordinates between detected FDI teeth
    to produce:
    1. 32-Tooth permanent dentition status list.
    2. Missing edentulous spaces with exact boundary coordinates.
    """
    h, w = img_shape[:2]
    detected_by_fdi = {t.fdi: t for t in detected_teeth}
    
    tooth_status_entries = []
    edentulous_gaps = []
    gap_id = 1

    # Check each quadrant sequence
    for q_name, seq in QUADRANTS.items():
        is_upper = (q_name in ("Q1", "Q2"))
        
        # Track active missing run in this quadrant
        current_missing_run = []
        last_present_tooth = None

        for idx, fdi in enumerate(seq):
            if fdi in detected_by_fdi:
                curr_tooth = detected_by_fdi[fdi]
                
                # If we had an active missing run bounded by last_present_tooth and curr_tooth
                if current_missing_run and last_present_tooth is not None:
                    # Calculate gap coordinates between last_present_tooth and curr_tooth
                    # For Q1 / Q4 (Right quadrant, midline to right):
                    # For Q2 / Q3 (Left quadrant, midline to left):
                    x_left = min(last_present_tooth.bbox[2], curr_tooth.bbox[0])
                    x_right = max(last_present_tooth.bbox[2], curr_tooth.bbox[0])
                    gap_w_px = max(x_right - x_left, 10)
                    gap_w_mm = gap_w_px / px_per_mm

                    crest_y = (last_present_tooth.cy + curr_tooth.cy) / 2.0

                    if gap_w_mm >= 4.0:
                        edentulous_gaps.append({
                            "gap_id": gap_id,
                            "missing_fdis": list(current_missing_run),
                            "adjacent_left_fdi": min(last_present_tooth.fdi, curr_tooth.fdi),
                            "adjacent_right_fdi": max(last_present_tooth.fdi, curr_tooth.fdi),
                            "bbox": (int(x_left), int(crest_y - 25), int(x_right), int(crest_y + 25)),
                            "center_x": (x_left + x_right) / 2.0,
                            "center_y": crest_y,
                            "width_px": float(gap_w_px),
                            "width_mm": round(gap_w_mm, 1),
                            "is_upper": is_upper,
                            "arch_region": curr_tooth.arch_region,
                        })
                        gap_id += 1

                current_missing_run = []
                last_present_tooth = curr_tooth

                tooth_status_entries.append({
                    "fdi": fdi,
                    "arch_region": curr_tooth.arch_region,
                    "detected": True,
                    "status": "Existing tooth",
                    "confidence": curr_tooth.confidence,
                    "bbox": curr_tooth.bbox,
                    "available_space_mm": round(curr_tooth.width_px / px_per_mm, 1),
                    "implant_consideration": "No",
                    "reason": f"Tooth FDI {fdi} detected (Confidence: {curr_tooth.confidence:.0%})",
                })
            else:
                # Tooth not detected -> record missing
                current_missing_run.append(fdi)
                
                # Check if third molar
                if fdi in (18, 28, 38, 48):
                    tooth_status_entries.append({
                        "fdi": fdi,
                        "arch_region": "Third Molar",
                        "detected": False,
                        "status": "Uncertain",
                        "confidence": 0.70,
                        "bbox": None,
                        "available_space_mm": None,
                        "implant_consideration": "No",
                        "reason": "Third molar not observed in dental arch (absent, unerupted, or extracted).",
                    })
                else:
                    tooth_status_entries.append({
                        "fdi": fdi,
                        "arch_region": "Dental Arch",
                        "detected": False,
                        "status": "Missing",
                        "confidence": 0.90,
                        "bbox": None,
                        "available_space_mm": None,
                        "implant_consideration": "Yes — Candidate Site",
                        "reason": f"FDI {fdi} missing in dental sequence. Edentulous site may be evaluated for implant replacement.",
                    })

    # Sort status entries by FDI
    status_map = {e["fdi"]: e for e in tooth_status_entries}
    ordered_status = [status_map[fdi] for fdi in ALL_32_FDI if fdi in status_map]

    return ordered_status, edentulous_gaps
