"""
train.py — training, validation and testing for both models

    --model 2d : 2D U-Net baseline on the coronal slices (keras_slices_data)
    --model 3d : 3D Improved U-Net on the full volumes

Both use 0.5 Dice + 0.5 cross-entropy over all 6 labels, Adam, and
ReduceLROnPlateau on the validation PROSTATE Dice (accumulated over the whole
validation split). The best checkpoint is chosen by validation prostate Dice.
Defaults are the settings used for the reported runs (see DEFAULTS).

Reported per split:
  - global per-class Dice (counts accumulated over the split, then one Dice)
  - mean per-sample prostate Dice (per slice in 2D, per volume in 3D) over
    samples that contain prostate
  - false-positive sample rate (2D: prostate-free slices with predicted prostate)

The 3D test numbers here are on the training grid; the official
native-resolution numbers and the base/apex analysis come from predict.py.

Examples:
    python train.py --model 2d --data_root .../keras_slices_data \
        --save_dir .../checkpoints_2d_seed1 --amp --seed 1
    python train.py --model 3d --image_dir .../semantic_MRs --mask_dir .../semantic_labels_only \
        --cache_dir /content/cache3d --save_dir .../checkpoints_3d_seed1 --amp --seed 1
"""

import argparse
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from dataset import NUM_CLASSES, PROSTATE_LABEL, SHAPE_3D, get_dataloaders_2d, get_dataloaders_3d
from modules import (UNet2D, ImprovedUNet3D, DiceCELoss, dice_counts, counts_to_dice,
                     per_sample_dice)

DEFAULTS = {
    "2d": dict(batch_size=16, epochs=20, lr=1e-3, base_filters=64, sched_patience=5,
               patience=0, grad_clip=0.0, amp_dtype="fp16"),
    "3d": dict(batch_size=2, epochs=150, lr=5e-4, base_filters=16, sched_patience=8,
               patience=25, grad_clip=1.0, amp_dtype="bf16"),
}


def build_model(model_type: str, base_filters: int, dropout: float = 0.3):
    if model_type == "2d":
        return UNet2D(1, NUM_CLASSES, base_filters)
    return ImprovedUNet3D(1, NUM_CLASSES, base_filters, dropout)


def autocast(device, amp_dtype):
    return torch.autocast(device.type, dtype=amp_dtype or torch.bfloat16,
                          enabled=amp_dtype is not None)


def train_one_epoch(model, loader, criterion, optimiser, device, scaler, amp_dtype, grad_clip):
    model.train()
    total, n = 0.0, 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimiser.zero_grad(set_to_none=True)
        with autocast(device, amp_dtype):
            loss = criterion(model(images), masks)
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimiser)
        scaler.update()
        total += loss.item() * images.size(0)
        n += images.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_dtype, label=PROSTATE_LABEL):
    model.eval()
    total, n = 0.0, 0
    inter = torch.zeros(NUM_CLASSES, device=device)
    pred_sum = torch.zeros(NUM_CLASSES, device=device)
    target_sum = torch.zeros(NUM_CLASSES, device=device)
    sample_dice, empty, fp = [], 0, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        with autocast(device, amp_dtype):
            logits = model(images)
            total += criterion(logits, masks).item() * images.size(0)
        logits = logits.float()
        n += images.size(0)
        i, p, t = dice_counts(logits, masks, NUM_CLASSES)
        inter, pred_sum, target_sum = inter + i, pred_sum + p, target_sum + t
        d, gt_present, pred_present = per_sample_dice(logits, masks, label)
        sample_dice.append(d[gt_present])
        empty += (~gt_present).sum().item()
        fp += (~gt_present & pred_present).sum().item()
    per_class = counts_to_dice(inter, pred_sum, target_sum).tolist()
    sd = torch.cat(sample_dice)
    return total / max(n, 1), {
        "per_class_dice": per_class,
        "prostate_dice": per_class[label],
        "sample_mean": sd.mean().item() if sd.numel() else 0.0,
        "sample_min": sd.min().item() if sd.numel() else 0.0,
        "fp_rate": fp / empty if empty else float("nan"),
    }


def format_metrics(m, unit):
    per_class = " ".join(f"c{c}:{d:.3f}" for c, d in enumerate(m["per_class_dice"]))
    return (f"Prostate Dice {m['prostate_dice']:.4f} | per-{unit} mean {m['sample_mean']:.4f} "
            f"(min {m['sample_min']:.3f}) | FP-{unit} rate {m['fp_rate']:.3f} | {per_class}")


def plot_curves(history, save_dir, target):
    epochs = range(1, len(history["train"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, history["train"], label="Train Loss")
    ax1.plot(epochs, history["val"], label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Training and Validation Loss")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["dice"], color="green", label="Val Prostate Dice (global)")
    ax2.axhline(target, color="grey", linestyle="--", label=f"Target {target:.2f}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Dice"); ax2.set_ylim(0, 1)
    ax2.set_title("Validation Prostate Dice"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Train the 2D U-Net or the 3D Improved U-Net")
    ap.add_argument("--model", choices=["2d", "3d"], required=True)
    ap.add_argument("--data_root", help="2D: keras_slices_data folder")
    ap.add_argument("--image_dir", help="3D: semantic_MRs folder")
    ap.add_argument("--mask_dir", help="3D: semantic_labels_only folder")
    ap.add_argument("--cache_dir", default=None, help="3D: cache for preprocessed volumes")
    ap.add_argument("--target_shape", type=int, nargs=3, default=list(SHAPE_3D),
                    help="3D: training grid D H W (default native 128 256 256)")
    ap.add_argument("--save_dir", default=None)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--base_filters", type=int)
    ap.add_argument("--dropout", type=float, default=0.3, help="3D context-module dropout")
    ap.add_argument("--sched_patience", type=int, help="ReduceLROnPlateau patience")
    ap.add_argument("--patience", type=int, help="Early-stopping patience (0 = off)")
    ap.add_argument("--grad_clip", type=float, help="Max grad norm (0 = off)")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--amp", action="store_true", help="Mixed precision on GPU")
    ap.add_argument("--amp_dtype", choices=["fp16", "bf16"])
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    for k, v in DEFAULTS[args.model].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    args.save_dir = args.save_dir or f"./checkpoints_{args.model}"

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    amp_dtype = None
    if args.amp and device.type == "cuda":
        amp_dtype = torch.float16
        if args.amp_dtype == "bf16" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
    print(f"Model: {args.model} | device: {gpu} | AMP: {amp_dtype} | seed: {args.seed}")

    if args.model == "2d":
        if not args.data_root:
            ap.error("--model 2d needs --data_root")
        loaders = get_dataloaders_2d(args.data_root, args.batch_size, args.num_workers,
                                     augment=not args.no_augment)
        unit, target = "slice", 0.75
    else:
        if not (args.image_dir and args.mask_dir):
            ap.error("--model 3d needs --image_dir and --mask_dir")
        loaders = get_dataloaders_3d(args.image_dir, args.mask_dir, args.batch_size,
                                     args.num_workers, augment=not args.no_augment,
                                     target_shape=tuple(args.target_shape),
                                     cache_dir=args.cache_dir)
        unit, target = "volume", 0.70
    train_loader, val_loader, test_loader = loaders

    model = build_model(args.model, args.base_filters, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    criterion = DiceCELoss()
    optimiser = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="max", factor=0.5,
                                                     patience=args.sched_patience)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    ckpt_path = os.path.join(args.save_dir, "best_model.pth")
    history = {"train": [], "val": [], "dice": []}
    best, best_epoch, stale = -1.0, 0, 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, device,
                                     scaler, amp_dtype, args.grad_clip)
        val_loss, m = evaluate(model, val_loader, criterion, device, amp_dtype)
        scheduler.step(m["prostate_dice"])
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["dice"].append(m["prostate_dice"])
        print(f"Epoch {epoch:3d}/{args.epochs} | Train {train_loss:.4f} | Val {val_loss:.4f} | "
              f"{format_metrics(m, unit)} | lr {optimiser.param_groups[0]['lr']:.1e} | "
              f"{time.time() - t0:.1f}s")

        if m["prostate_dice"] > best:
            best, best_epoch, stale = m["prostate_dice"], epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "best_dice": best, "args": vars(args),
                        "base_filters": args.base_filters,
                        "prostate_label": PROSTATE_LABEL}, ckpt_path)
            print(f"  -> Best model saved (val prostate Dice {best:.4f})")
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs")
                break

    minutes = (time.time() - start) / 60
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0
    print(f"\nTraining done in {minutes:.1f} min, peak VRAM {peak:.2f} GB")
    plot_curves(history, args.save_dir, target)

    model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                     weights_only=True)["model_state_dict"])
    test_loss, tm = evaluate(model, test_loader, criterion, device, amp_dtype)
    print(f"\n--- Test --- Loss {test_loss:.4f} | {format_metrics(tm, unit)}")

    with open(os.path.join(args.save_dir, "test_results.txt"), "w") as f:
        if args.model == "3d":
            f.write(f"Grid {args.target_shape}; native-resolution results come from predict.py\n")
        f.write(f"Best epoch: {best_epoch}\nBest val prostate Dice: {best:.4f}\n")
        f.write(f"Test loss: {test_loss:.4f}\n")
        f.write(f"Test prostate Dice (global): {tm['prostate_dice']:.4f}\n")
        f.write(f"Test prostate Dice (per-{unit} mean / min, {unit}s with prostate): "
                f"{tm['sample_mean']:.4f} / {tm['sample_min']:.4f}\n")
        f.write(f"Test FP-{unit} rate: {tm['fp_rate']:.4f}\n")
        f.write("Per-class test Dice (global):\n")
        for c, d in enumerate(tm["per_class_dice"]):
            f.write(f"  class {c}: {d:.4f}\n")
        f.write(f"Parameters: {n_params:,}\nTraining time: {minutes:.1f} min\n")
        f.write(f"Peak training VRAM: {peak:.2f} GB\nGPU: {gpu}\n")
        f.write(f"Batch size: {args.batch_size}, epochs: {args.epochs}, lr: {args.lr}, "
                f"AMP: {amp_dtype}, seed: {args.seed}\n")


if __name__ == "__main__":
    main()
