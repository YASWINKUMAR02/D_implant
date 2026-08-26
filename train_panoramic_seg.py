"""
train_panoramic_seg.py
======================
2D Teeth Segmentation from Panoramic X-rays (OPG)
Dataset : Panoramic_Dental_Xray_Segmentation_Dataset (329 images)
Model   : U-Net++ with ResNet34 encoder (ImageNet pretrained)
Task    : Binary segmentation — Background (0) vs Teeth (1)
Loss    : BCE-Dice Combined (handles 7.2:1 class imbalance)
Device  : CUDA (RTX 2050) / CPU fallback

Usage:
    python train_panoramic_seg.py               # train with defaults
    python train_panoramic_seg.py --epochs 100  # custom epochs
    python train_panoramic_seg.py --predict path/to/image.jpg  # run inference
"""

import os
import sys
import glob
import time
import random
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import CosineAnnealingLR
import csv

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "Panoramic_Dental_Xray_Segmentation_Dataset"
IMG_DIR     = DATASET_DIR / "images"
MASK_DIR    = DATASET_DIR / "masks"
MODELS_DIR  = ROOT_DIR / "models"
LOGS_DIR    = ROOT_DIR / "outputs" / "panoramic_seg_logs"

MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

CFG = {
    "img_height"     : 512,
    "img_width"      : 1024,
    "in_channels"    : 1,
    "num_classes"    : 1,
    "encoder"        : "resnet34",
    "encoder_weights": "imagenet",
    "batch_size"     : 4,
    "num_epochs"     : 100,
    "lr"             : 1e-4,
    "weight_decay"   : 1e-4,
    "val_split"      : 0.10,
    "test_split"     : 0.10,
    "seed"           : 42,
    "num_workers"    : 2,
    "dice_weight"    : 0.5,
    "bce_weight"     : 0.5,
    "pos_weight"     : 7.2,
    "early_stop"     : 15,
    "checkpoint"     : str(MODELS_DIR / "panoramic_teeth_seg.pth"),
    "best_model"     : str(MODELS_DIR / "panoramic_teeth_seg_best.pth"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOGS_DIR / "train.log")),
    ]
)
log = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class PanoramicTeethDataset(Dataset):
    """
    Panoramic dental X-ray segmentation dataset.
    Images : JPG BGR 3-ch -> grayscale 1-ch
    Masks  : PNG uint8 {0=background, 1=teeth}
    """

    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = list(image_paths)
        self.mask_paths  = list(mask_paths)
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load & convert image to grayscale
        img_bgr  = cv2.imread(str(self.image_paths[idx]))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read: {self.image_paths[idx]}")
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # (H, W) uint8

        # Load mask
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {self.mask_paths[idx]}")
        mask = (mask > 0).astype(np.uint8)  # ensure binary {0,1}

        if self.transform:
            aug      = self.transform(image=img_gray, mask=mask)
            img_gray = aug["image"]  # (1, H, W) tensor
            mask     = aug["mask"]   # (H, W) tensor

        return img_gray, mask.float()

    @staticmethod
    def get_paired_paths(img_dir, mask_dir):
        img_paths  = sorted(Path(img_dir).glob("*.jpg")) + sorted(Path(img_dir).glob("*.png"))
        valid_imgs, mask_paths = [], []
        for ip in img_paths:
            mp = Path(mask_dir) / f"{ip.stem}_mask.png"
            if mp.exists():
                valid_imgs.append(ip)
                mask_paths.append(mp)
            else:
                log.warning(f"No mask for {ip.name} — skipped")
        log.info(f"Found {len(valid_imgs)} valid image-mask pairs")
        return valid_imgs, mask_paths


# ─────────────────────────────────────────────────────────────────────────────
# Augmentations
# ─────────────────────────────────────────────────────────────────────────────
def get_train_transforms(h, w):
    return A.Compose([
        A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=5, border_mode=cv2.BORDER_REFLECT, p=0.5),
        A.ElasticTransform(alpha=60, sigma=8, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.0, 0.05), p=0.2),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.Normalize(mean=(0.485,), std=(0.229,)),
        ToTensorV2(),
    ])


def get_val_transforms(h, w):
    return A.Compose([
        A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=(0.485,), std=(0.229,)),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Loss: BCE + Dice
# ─────────────────────────────────────────────────────────────────────────────
class BCEDiceLoss(nn.Module):
    """Combined BCE (with pos_weight) + Dice Loss for class imbalance."""

    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=7.2, smooth=1.0):
        super().__init__()
        self.bce_w   = bce_weight
        self.dice_w  = dice_weight
        self.smooth  = smooth
        self.bce_fn  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, logits, targets):
        if self.bce_fn.pos_weight.device != logits.device:
            self.bce_fn.pos_weight = self.bce_fn.pos_weight.to(logits.device)

        bce_loss = self.bce_fn(logits, targets)

        probs  = torch.sigmoid(logits).view(-1)
        tgts   = targets.view(-1)
        inter  = (probs * tgts).sum()
        dice_loss = 1.0 - (2.0 * inter + self.smooth) / (probs.sum() + tgts.sum() + self.smooth)

        return self.bce_w * bce_loss + self.dice_w * dice_loss


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(logits, targets, threshold=0.5):
    preds  = (torch.sigmoid(logits) > threshold).float().view(-1)
    tgts   = targets.view(-1)
    tp = (preds * tgts).sum()
    fp = (preds * (1 - tgts)).sum()
    fn = ((1 - preds) * tgts).sum()
    eps = 1e-6
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou  = (tp + eps) / (tp + fp + fn + eps)
    return {"dice": dice.item(), "iou": iou.item()}


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
def build_model(cfg):
    model = smp.UnetPlusPlus(
        encoder_name    = cfg["encoder"],
        encoder_weights = cfg["encoder_weights"],
        in_channels     = cfg["in_channels"],
        classes         = cfg["num_classes"],
        activation      = None,
    )
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: U-Net++ | Encoder: {cfg['encoder']} | Params: {n_params:,}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Training / Validation loops
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    loss_sum, dice_sum, iou_sum = 0.0, 0.0, 0.0
    nb = len(loader)
    log_every = max(1, nb // 4)

    for i, (imgs, masks) in enumerate(loader, 1):
        imgs  = imgs.to(device, dtype=torch.float32)
        masks = masks.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        m = compute_metrics(logits.detach(), masks)
        loss_sum += loss.item()
        dice_sum += m["dice"]
        iou_sum  += m["iou"]

        if i % log_every == 0:
            log.info(f"  Ep{epoch} [{i}/{nb}] Loss={loss.item():.4f} Dice={m['dice']:.4f} IoU={m['iou']:.4f}")

    return {"loss": loss_sum/nb, "dice": dice_sum/nb, "iou": iou_sum/nb}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_sum, dice_sum, iou_sum = 0.0, 0.0, 0.0
    nb = len(loader)
    for imgs, masks in loader:
        imgs  = imgs.to(device, dtype=torch.float32)
        masks = masks.to(device).unsqueeze(1)
        logits = model(imgs)
        loss   = criterion(logits, masks)
        m      = compute_metrics(logits, masks)
        loss_sum += loss.item()
        dice_sum += m["dice"]
        iou_sum  += m["iou"]
    return {"loss": loss_sum/nb, "dice": dice_sum/nb, "iou": iou_sum/nb}


# ─────────────────────────────────────────────────────────────────────────────
# Main train function
# ─────────────────────────────────────────────────────────────────────────────
def train(cfg):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    log.info(f"Device: {device} ({gpu_name})")

    # ── Paths ─────────────────────────────────────────────────────────────────
    all_imgs, all_masks = PanoramicTeethDataset.get_paired_paths(IMG_DIR, MASK_DIR)
    n = len(all_imgs)

    # ── Split ─────────────────────────────────────────────────────────────────
    n_test  = int(n * cfg["test_split"])
    n_val   = int(n * cfg["val_split"])
    n_train = n - n_val - n_test
    idx     = list(range(n))
    random.shuffle(idx)
    tr_idx, va_idx, te_idx = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]
    log.info(f"Split → Train:{len(tr_idx)} | Val:{len(va_idx)} | Test:{len(te_idx)}")

    tr_imgs  = [all_imgs[i]  for i in tr_idx]
    tr_masks = [all_masks[i] for i in tr_idx]
    va_imgs  = [all_imgs[i]  for i in va_idx]
    va_masks = [all_masks[i] for i in va_idx]
    te_imgs  = [all_imgs[i]  for i in te_idx]
    te_masks = [all_masks[i] for i in te_idx]

    # Save test split
    np.save(str(LOGS_DIR / "test_imgs.npy"),  [str(p) for p in te_imgs])
    np.save(str(LOGS_DIR / "test_masks.npy"), [str(p) for p in te_masks])

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_tf = get_train_transforms(cfg["img_height"], cfg["img_width"])
    val_tf   = get_val_transforms(cfg["img_height"], cfg["img_width"])

    train_ds = PanoramicTeethDataset(tr_imgs, tr_masks, transform=train_tf)
    val_ds   = PanoramicTeethDataset(va_imgs, va_masks, transform=val_tf)
    test_ds  = PanoramicTeethDataset(te_imgs, te_masks, transform=val_tf)

    train_ld = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=cfg["num_workers"], pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                          num_workers=cfg["num_workers"], pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False,
                          num_workers=cfg["num_workers"], pin_memory=True)

    # ── Model / Loss / Optimizer / Scheduler ──────────────────────────────────
    model     = build_model(cfg).to(device)
    criterion = BCEDiceLoss(cfg["bce_weight"], cfg["dice_weight"], cfg["pos_weight"])
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"], eta_min=1e-6)
    # CSV metrics logger (no TensorBoard dependency)
    csv_path = LOGS_DIR / "metrics.csv"
    csv_file = open(str(csv_path), "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "tr_loss", "tr_dice", "tr_iou", "val_loss", "val_dice", "val_iou", "lr"])

    best_dice  = 0.0
    no_improve = 0
    t0         = time.time()

    log.info("=" * 62)
    log.info(" Starting Training — U-Net++ Panoramic Teeth Segmentation")
    log.info("=" * 62)

    for epoch in range(1, cfg["num_epochs"] + 1):
        t_ep = time.time()

        tr  = train_one_epoch(model, train_ld, optimizer, criterion, device, epoch)
        val = validate(model, val_ld, criterion, device)
        scheduler.step()

        elapsed = time.time() - t_ep
        lr_now  = scheduler.get_last_lr()[0]
        log.info(
            f"Ep {epoch:03d}/{cfg['num_epochs']} [{elapsed:.0f}s] "
            f"| TR  Loss={tr['loss']:.4f} Dice={tr['dice']:.4f} IoU={tr['iou']:.4f} "
            f"| VAL Loss={val['loss']:.4f} Dice={val['dice']:.4f} IoU={val['iou']:.4f} "
            f"| LR={lr_now:.2e}"
        )

        csv_writer.writerow([epoch, f"{tr['loss']:.4f}", f"{tr['dice']:.4f}", f"{tr['iou']:.4f}",
                              f"{val['loss']:.4f}", f"{val['dice']:.4f}", f"{val['iou']:.4f}", f"{lr_now:.2e}"])
        csv_file.flush()

        ckpt = {
            "epoch": epoch, "val_dice": val["dice"], "cfg": cfg,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }
        torch.save(ckpt, cfg["checkpoint"])

        if val["dice"] > best_dice:
            best_dice  = val["dice"]
            no_improve = 0
            torch.save(ckpt, cfg["best_model"])
            log.info(f"  ✅ New best! Val Dice={best_dice:.4f}  Saved → {cfg['best_model']}")
        else:
            no_improve += 1
            log.info(f"  No improvement {no_improve}/{cfg['early_stop']}")

        if no_improve >= cfg["early_stop"]:
            log.info(f"Early stopping at epoch {epoch}")
            break

    csv_file.close()
    log.info(f"Metrics saved → {csv_path}")
    total = (time.time() - t0) / 60
    log.info(f"\nTraining done in {total:.1f} min | Best Val Dice: {best_dice:.4f}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    log.info("Evaluating on test set...")
    best_ckpt = torch.load(cfg["best_model"], map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    te = validate(model, test_ld, criterion, device)
    log.info(f"TEST → Loss={te['loss']:.4f}  Dice={te['dice']:.4f}  IoU={te['iou']:.4f}")

    return model, best_dice


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def predict(image_path, model_path=None, threshold=0.5, save_output=True):
    """Run inference on a single panoramic X-ray. Returns binary mask (uint8 0/255)."""
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = build_model(CFG).to(device)
    ckpt_path = model_path or CFG["best_model"]
    ckpt      = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info(f"Loaded model from {ckpt_path} (epoch {ckpt.get('epoch','?')})")

    img_bgr  = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    orig_h, orig_w = img_bgr.shape[:2]
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    tf      = get_val_transforms(CFG["img_height"], CFG["img_width"])
    aug     = tf(image=img_gray, mask=np.zeros((orig_h, orig_w), np.uint8))
    tensor  = aug["image"].unsqueeze(0).float().to(device)

    with torch.no_grad():
        logit = model(tensor)
        prob  = torch.sigmoid(logit).squeeze().cpu().numpy()

    pred_mask = (prob > threshold).astype(np.uint8) * 255
    pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Green overlay
    overlay    = img_bgr.copy()
    green      = np.zeros_like(img_bgr)
    green[pred_mask > 0] = [0, 220, 80]
    overlay    = cv2.addWeighted(overlay, 0.7, green, 0.3, 0)

    if save_output:
        out = ROOT_DIR / "outputs" / "predictions"
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        cv2.imwrite(str(out / f"{stem}_mask.png"),    pred_mask)
        cv2.imwrite(str(out / f"{stem}_overlay.png"), overlay)
        log.info(f"Saved → {out}/{stem}_mask.png + _overlay.png")

    return pred_mask


# ─────────────────────────────────────────────────────────────────────────────
# Visualize predictions (optional)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def visualize_predictions(model, dataset, device, n=4, save_dir=None):
    save_dir = save_dir or (LOGS_DIR / "vis")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.eval()

    for i, idx in enumerate(random.sample(range(len(dataset)), min(n, len(dataset)))):
        img_t, mask_t = dataset[idx]
        logit = model(img_t.unsqueeze(0).float().to(device))
        pred  = (torch.sigmoid(logit).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        mask  = mask_t.numpy().astype(np.uint8)
        img   = img_t.squeeze().numpy()

        # Denormalize
        img_disp = np.clip((img * 0.229 + 0.485) * 255, 0, 255).astype(np.uint8)
        img_col  = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2BGR)
        gt_col   = np.stack([mask * 255] * 3, -1).astype(np.uint8)
        pred_col = np.stack([pred * 255] * 3, -1).astype(np.uint8)

        panel = np.hstack([img_col, gt_col, pred_col])
        W = img_disp.shape[1]
        cv2.putText(panel, "Input",        (10,   30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        cv2.putText(panel, "Ground Truth", (W+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),   2)
        cv2.putText(panel, "Prediction",  (2*W+10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,120,255), 2)
        cv2.imwrite(str(Path(save_dir) / f"vis_{i:02d}.png"), panel)

    log.info(f"Saved {n} visualizations → {save_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Panoramic Teeth Segmentation — U-Net++")
    parser.add_argument("--epochs",     type=int,   default=CFG["num_epochs"])
    parser.add_argument("--batch",      type=int,   default=CFG["batch_size"])
    parser.add_argument("--lr",         type=float, default=CFG["lr"])
    parser.add_argument("--encoder",    type=str,   default=CFG["encoder"],
                        help="resnet34 | resnet50 | efficientnet-b4")
    parser.add_argument("--img_h",      type=int,   default=CFG["img_height"])
    parser.add_argument("--img_w",      type=int,   default=CFG["img_width"])
    parser.add_argument("--predict",    type=str,   default=None,
                        help="Path to X-ray image for inference (skips training)")
    parser.add_argument("--model_path", type=str,   default=None)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--visualize",  action="store_true",
                        help="Save sample prediction visualizations after training")
    args = parser.parse_args()

    CFG.update({
        "num_epochs": args.epochs,
        "batch_size": args.batch,
        "lr":         args.lr,
        "encoder":    args.encoder,
        "img_height": args.img_h,
        "img_width":  args.img_w,
    })

    if args.predict:
        log.info(f"== Inference mode: {args.predict} ==")
        mask = predict(args.predict, args.model_path, args.threshold)
        log.info(f"Teeth pixels: {(mask > 0).sum()}")
        sys.exit(0)

    log.info("=" * 62)
    log.info(" U-Net++ Panoramic Teeth Segmentation — Training")
    log.info("=" * 62)
    log.info(f"  Encoder : {CFG['encoder']}   Input: {CFG['img_width']}x{CFG['img_height']}")
    log.info(f"  Epochs  : {CFG['num_epochs']}   Batch: {CFG['batch_size']}   LR: {CFG['lr']}")
    log.info("=" * 62)

    model, best_dice = train(CFG)

    if args.visualize:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        val_tf = get_val_transforms(CFG["img_height"], CFG["img_width"])
        all_i, all_m = PanoramicTeethDataset.get_paired_paths(IMG_DIR, MASK_DIR)
        full_ds = PanoramicTeethDataset(all_i, all_m, transform=val_tf)
        visualize_predictions(model, full_ds, device, n=6)

    log.info("Done. ✅")
