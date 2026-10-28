"""
train.py — Training pipeline for the 3D Improved U-Net on HipMRI volumes

  - Same loss as the 2D baseline (0.5 Dice + 0.5 CE over all 6 labels)
  - Adam, ReduceLROnPlateau and early stopping driven by validation
    PROSTATE Dice (accumulated over the whole validation split)
  - Mixed precision: bf16 on GPUs that support it (A100), else fp16
  - Test numbers printed here are on the resampled 128^3 grid; the official
    native-resolution numbers and the base/apex analysis come from predict.py

Usage:
    python train.py --image_dir .../semantic_MRs --mask_dir .../semantic_labels_only \
        --cache_dir /content/cache3d --save_dir .../checkpoints/unet3d_seed1 --amp --seed 1
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules import ImprovedUNet3D, DiceCELoss, dice_counts, counts_to_dice, per_sample_dice
from dataset import get_dataloaders, NUM_CLASSES


def train_one_epoch(model, loader, criterion, optimiser, device, scaler, amp_dtype,
                    grad_clip):
    model.train()
    total, n = 0.0, 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimiser.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=amp_dtype or torch.bfloat16,
                            enabled=amp_dtype is not None):
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
def evaluate(model, loader, criterion, device, prostate_label, amp_dtype):
    model.eval()
    total, n = 0.0, 0
    inter = torch.zeros(NUM_CLASSES, device=device)
    pred_sum = torch.zeros(NUM_CLASSES, device=device)
    target_sum = torch.zeros(NUM_CLASSES, device=device)
    vol_dices = []
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        with torch.autocast(device.type, dtype=amp_dtype or torch.bfloat16,
                            enabled=amp_dtype is not None):
            logits = model(images)
            total += criterion(logits, masks).item() * images.size(0)
        logits = logits.float()
        n += images.size(0)
        i, p, t = dice_counts(logits, masks, NUM_CLASSES)
        inter, pred_sum, target_sum = inter + i, pred_sum + p, target_sum + t
        vol_dices.append(per_sample_dice(logits, masks, prostate_label))
    per_class = counts_to_dice(inter, pred_sum, target_sum).tolist()
    vol = torch.cat(vol_dices)
    return total / max(n, 1), {
        "per_class_dice": per_class,
        "prostate_dice": per_class[prostate_label],
        "prostate_volume_mean": vol.mean().item(),
        "prostate_volume_min": vol.min().item(),
    }


def format_metrics(m):
    per_class = " ".join(f"c{c}:{d:.3f}" for c, d in enumerate(m["per_class_dice"]))
    return (f"Prostate Dice {m['prostate_dice']:.4f} | per-volume mean "
            f"{m['prostate_volume_mean']:.4f} (min {m['prostate_volume_min']:.3f}) | {per_class}")


def plot_curves(history, save_dir):
    epochs = range(1, len(history["train"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, history["train"], label="Train Loss")
    ax1.plot(epochs, history["val"], label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_title("Training and Validation Loss")
    ax2.plot(epochs, history["dice"], color="green", label="Val Prostate Dice (global)")
    ax2.axhline(0.7, color="grey", linestyle="--", label="Target 0.70")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Dice"); ax2.set_ylim(0, 1)
    ax2.legend(); ax2.grid(alpha=0.3); ax2.set_title("Validation Prostate Dice")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Train 3D Improved U-Net on HipMRI")
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--cache_dir", default=None,
                    help="Folder for resampled volumes (fast local disk, e.g. /content/cache3d)")
    ap.add_argument("--save_dir", default="./checkpoints")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--base_filters", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--target_shape", type=int, nargs=3, default=[128, 128, 128])
    ap.add_argument("--prostate_label", type=int, default=5)
    ap.add_argument("--lr_axis", type=int, default=1,
                    help="Left-right axis of the (D, H, W) volume, used for flipping")
    ap.add_argument("--patience", type=int, default=25, help="Early-stopping patience (epochs)")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

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
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Device: {gpu} | AMP: {amp_dtype} | seed: {args.seed}")

    train_loader, val_loader, test_loader = get_dataloaders(
        args.image_dir, args.mask_dir, args.batch_size, args.num_workers,
        augment=not args.no_augment, target_shape=tuple(args.target_shape),
        cache_dir=args.cache_dir, lr_axis=args.lr_axis)

    model = ImprovedUNet3D(1, NUM_CLASSES, args.base_filters, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    criterion = DiceCELoss()
    optimiser = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="max",
                                                     factor=0.5, patience=8)
    # GradScaler is only needed for fp16; it is a no-op otherwise
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history = {"train": [], "val": [], "dice": []}
    best, best_epoch, stale = -1.0, 0, 0
    start = time.time()
    ckpt_path = os.path.join(args.save_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, device,
                                     scaler, amp_dtype, args.grad_clip)
        val_loss, m = evaluate(model, val_loader, criterion, device,
                               args.prostate_label, amp_dtype)
        scheduler.step(m["prostate_dice"])
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["dice"].append(m["prostate_dice"])
        print(f"Epoch {epoch:3d} | Train {train_loss:.4f} | Val {val_loss:.4f} | "
              f"{format_metrics(m)} | lr {optimiser.param_groups[0]['lr']:.1e} | "
              f"{time.time() - t0:.1f}s")

        if m["prostate_dice"] > best:
            best, best_epoch, stale = m["prostate_dice"], epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "best_dice": best, "args": vars(args)}, ckpt_path)
            print(f"  -> Best model saved (val prostate Dice {best:.4f})")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs")
                break

    minutes = (time.time() - start) / 60
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0
    print(f"\nTraining done in {minutes:.1f} min, peak VRAM {peak:.2f} GB")
    plot_curves(history, args.save_dir)

    model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                     weights_only=True)["model_state_dict"])
    test_loss, tm = evaluate(model, test_loader, criterion, device,
                             args.prostate_label, amp_dtype)
    print(f"\n--- Test (128^3 grid) --- Loss {test_loss:.4f} | {format_metrics(tm)}")

    with open(os.path.join(args.save_dir, "test_results.txt"), "w") as f:
        f.write("3D Improved U-Net — test results on the resampled grid "
                "(native-resolution results: predict.py)\n")
        f.write(f"Best epoch: {best_epoch}\nBest val prostate Dice: {best:.4f}\n")
        f.write(f"Test prostate Dice (global): {tm['prostate_dice']:.4f}\n")
        f.write(f"Test prostate Dice (per-volume mean / min): "
                f"{tm['prostate_volume_mean']:.4f} / {tm['prostate_volume_min']:.4f}\n")
        for c, d in enumerate(tm["per_class_dice"]):
            f.write(f"  class {c}: {d:.4f}\n")
        f.write(f"Parameters: {n_params:,}\nTraining time: {minutes:.1f} min\n")
        f.write(f"Peak training VRAM: {peak:.2f} GB\nGPU: {gpu}\n")
        f.write(f"Batch size: {args.batch_size}, lr: {args.lr}, AMP: {amp_dtype}, "
                f"seed: {args.seed}, target shape: {args.target_shape}\n")


if __name__ == "__main__":
    main()
