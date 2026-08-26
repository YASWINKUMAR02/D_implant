"""
train_yolo_teeth.py
===================
Trains a YOLOv8 model for 32-Class FDI Tooth Detection & Numbering
using the dataset at: 'teeth detection and numbering.v18i.yolov7pytorch'

Outputs best weights to: 'models/yolov8_teeth_fdi_best.pt'
"""

import os
import shutil
from pathlib import Path
import yaml
from ultralytics import YOLO

ROOT_DIR = Path(__file__).parent.resolve()
DATASET_DIR = ROOT_DIR / "teeth detection and numbering.v18i.yolov7pytorch"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

def prepare_yaml():
    """Create an absolute-path data config yaml for reliable training."""
    dataset_yaml_path = DATASET_DIR / "data.yaml"
    with open(dataset_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Use absolute paths for robust resolution
    train_yaml_path = DATASET_DIR / "data_train.yaml"
    config = {
        "path": str(DATASET_DIR.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": data["nc"],
        "names": data["names"],
    }
    with open(train_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)
    
    print(f"Created training config: {train_yaml_path}")
    print(f"Classes: {config['nc']} teeth -> {config['names']}")
    return train_yaml_path


def train(epochs: int = 50, batch_size: int = 8, imgsz: int = 640):
    yaml_file = prepare_yaml()
    
    print(f"\n=======================================================")
    print(f" Starting YOLOv8 32-Tooth FDI Numbering Training")
    print(f" Dataset: {DATASET_DIR}")
    print(f" Epochs : {epochs} | Batch: {batch_size} | ImgSz: {imgsz}")
    print(f"=======================================================\n")
    
    # Initialize YOLOv8 nano / small pretrained backbone
    model = YOLO("yolov8n.pt")
    
    results = model.train(
        data=str(yaml_file),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=2,
        device=0,          # GPU 0
        project=str(ROOT_DIR / "runs" / "teeth_yolo"),
        name="fdi_32class",
        exist_ok=True,
        save=True,
        plots=True,
        patience=15,
    )
    
    # Copy best weights to models/
    best_weights = ROOT_DIR / "runs" / "teeth_yolo" / "fdi_32class" / "weights" / "best.pt"
    dest_weights = MODELS_DIR / "yolov8_teeth_fdi_best.pt"
    
    if best_weights.exists():
        shutil.copy(best_weights, dest_weights)
        print(f"\n[SUCCESS] Trained model saved to: {dest_weights}")
    else:
        print(f"\n[INFO] Training finished. Check runs/ directory for output weights.")

if __name__ == "__main__":
    train(epochs=40, batch_size=8, imgsz=640)
