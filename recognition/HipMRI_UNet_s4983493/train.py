"""
train.py — Training, Validation, and Testing Pipeline for 2D U-Net (baseline)

  - Combined Dice + Cross-Entropy loss over all 6 HipMRI labels
  - Adam optimiser, ReduceLROnPlateau driven by validation PROSTATE Dice
  - Best checkpoint selected by validation PROSTATE Dice (not mean over
    classes, which is dominated by background/body)
  - Reports, per split:
      * global per-class Dice (counts accumulated over the whole split)
      * mean per-slice prostate Dice over slices that contain prostate
      * false-positive slice rate: fraction of prostate-free slices where
        the model still predicts prostate
  - Saves loss / Dice curves and a text summary

Usage:
    python train.py --data_root /path/to/keras_slices_data --epochs 50
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules import UNet, DiceCELoss, dice_counts, counts_to_dice, per_slice_dice
from dataset import get_dataloaders, NUM_CLASSES


def train_one_epoch(model: nn.Module, loader, criterion, optimiser,
                    device: torch.device) -> float:
    """Train for one epoch. Returns average training loss."""
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimiser.zero_grad()
        loss = criterion(model(images), masks)
        loss.backward()
        optimiser.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device,
             num_classes: int, prostate_label: int) -> tuple[float, dict]:
    """Evaluate on a whole split.

    Returns:
        avg_loss, metrics dict with keys:
            per_class_dice    list[float], global Dice per class
            prostate_dice     global prostate Dice (headline metric)
            prostate_slice    mean per-slice prostate Dice (prostate slices only)
            fp_slice_rate     prostate predicted on prostate-free slices
    """
    model.eval()
    running_loss = 0.0
    inter = torch.zeros(num_classes, device=device)
    pred_sum = torch.zeros(num_classes, device=device)
    target_sum = torch.zeros(num_classes, device=device)
    slice_dices = []
    empty_slices, fp_slices = 0, 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        running_loss += criterion(logits, masks).item() * images.size(0)

        i, p, t = dice_counts(logits, masks, num_classes)
        inter += i
        pred_sum += p
        target_sum += t

        d, gt_present, pred_present = per_slice_dice(logits, masks, prostate_label)
        slice_dices.append(d[gt_present])
        empty = ~gt_present
        empty_slices += empty.sum().item()
        fp_slices += (empty & pred_present).sum().item()

    per_class = counts_to_dice(inter, pred_sum, target_sum).tolist()
    slice_cat = torch.cat(slice_dices)
    metrics = {
        "per_class_dice": per_class,
        "prostate_dice": per_class[prostate_label],
        "prostate_slice": slice_cat.mean().item() if slice_cat.numel() else 0.0,
        "fp_slice_rate": fp_slices / max(empty_slices, 1),
    }
    return running_loss / len(loader.dataset), metrics


def plot_training_curves(train_losses: list, val_losses: list,
                         val_prostate: list, save_dir: str):
    """Save loss curves and validation prostate Dice curve."""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_prostate, label="Val Prostate Dice (global)", color="green")
    ax2.axhline(0.75, color="grey", linestyle="--", label="Target 0.75")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice")
    ax2.set_title("Validation Prostate Dice")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()


def format_metrics(m: dict) -> str:
    per_class = " ".join(f"c{c}:{d:.3f}" for c, d in enumerate(m["per_class_dice"]))
    return (f"Prostate Dice {m['prostate_dice']:.4f} | "
            f"slice-mean {m['prostate_slice']:.4f} | "
            f"FP-slice rate {m['fp_slice_rate']:.3f} | {per_class}")


def main():
    parser = argparse.ArgumentParser(description="Train 2D U-Net baseline on HipMRI")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base_filters", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prostate_label", type=int, default=5,
                        help="Label value of the prostate in the masks")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--no_augment", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_root, args.batch_size, args.num_workers, augment=not args.no_augment)

    model = UNet(in_channels=1, num_classes=NUM_CLASSES,
                 base_filters=args.base_filters).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = DiceCELoss(dice_weight=0.5, ce_weight=0.5)
    optimiser = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=5)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_losses, val_losses, val_prostate = [], [], []
    best_dice = -1.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, device)
        val_loss, val_m = evaluate(model, val_loader, criterion, device,
                                   NUM_CLASSES, args.prostate_label)
        scheduler.step(val_m["prostate_dice"])

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_prostate.append(val_m["prostate_dice"])

        print(f"Epoch {epoch:3d}/{args.epochs} | Train {train_loss:.4f} | "
              f"Val {val_loss:.4f} | {format_metrics(val_m)} | "
              f"{time.time() - epoch_start:.1f}s")

        if val_m["prostate_dice"] > best_dice:
            best_dice = val_m["prostate_dice"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "best_dice": best_dice,
                "prostate_label": args.prostate_label,
                "base_filters": args.base_filters,
            }, os.path.join(args.save_dir, "best_model.pth"))
            print(f"  -> Best model saved (val prostate Dice {best_dice:.4f})")

    total_time = time.time() - start_time
    peak_vram = (torch.cuda.max_memory_allocated() / 1024 ** 3
                 if device.type == "cuda" else 0.0)
    print(f"\nTraining complete in {total_time / 60:.1f} min, "
          f"peak VRAM {peak_vram:.2f} GB")

    plot_training_curves(train_losses, val_losses, val_prostate, args.save_dir)

    # Final test evaluation with the best checkpoint
    checkpoint = torch.load(os.path.join(args.save_dir, "best_model.pth"),
                            map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_m = evaluate(model, test_loader, criterion, device,
                                 NUM_CLASSES, args.prostate_label)
    print(f"\n--- Test --- Loss {test_loss:.4f} | {format_metrics(test_m)}")

    with open(os.path.join(args.save_dir, "test_results.txt"), "w") as f:
        f.write(f"Best epoch: {checkpoint['epoch']}\n")
        f.write(f"Best val prostate Dice: {best_dice:.4f}\n")
        f.write(f"Test loss: {test_loss:.4f}\n")
        f.write(f"Test prostate Dice (global): {test_m['prostate_dice']:.4f}\n")
        f.write(f"Test prostate Dice (per-slice mean, prostate slices): "
                f"{test_m['prostate_slice']:.4f}\n")
        f.write(f"Test FP-slice rate: {test_m['fp_slice_rate']:.4f}\n")
        f.write("Per-class test Dice (global):\n")
        for c, d in enumerate(test_m["per_class_dice"]):
            f.write(f"  class {c}: {d:.4f}\n")
        f.write(f"Parameters: {total_params:,}\n")
        f.write(f"Training time: {total_time / 60:.1f} min\n")
        f.write(f"Peak training VRAM: {peak_vram:.2f} GB\n")


if __name__ == "__main__":
    main()
