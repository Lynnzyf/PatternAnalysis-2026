"""
predict.py — Inference and Visualisation for the Trained 2D U-Net

Loads a trained checkpoint, runs inference on the test set, prints Dice
metrics and per-slice latency, and saves multi-panel figures:
  - Input MRI slice
  - Ground truth segmentation
  - Predicted segmentation
  - Prostate difference overlay (TP green, FP red, FN blue)

Usage:
    python predict.py --data_root /path/to/keras_slices_data \
                      --checkpoint ./checkpoints/best_model.pth \
                      --output_dir ./predictions

NumPy is allowed in this file (per spec).
"""

import argparse
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from modules import UNet, dice_counts, counts_to_dice
from dataset import HipMRIDataset, NUM_CLASSES

SEG_COLORS = ListedColormap(
    ["black", "#7f8c8d", "#f1c40f", "#3498db", "#9b59b6", "#e74c3c"])
# Label names below 5 are assumed from previous-year conventions; verify.
CLASS_NAMES = {0: "Background", 1: "Body", 2: "Bone",
               3: "Bladder", 4: "Rectum", 5: "Prostate"}


def load_model(checkpoint_path: str, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Load trained U-Net and its checkpoint metadata."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = UNet(in_channels=1, num_classes=NUM_CLASSES,
                 base_filters=checkpoint.get("base_filters", 64)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded {checkpoint_path} (epoch {checkpoint.get('epoch', '?')}, "
          f"val prostate Dice {checkpoint.get('best_dice', float('nan')):.4f})")
    return model, checkpoint


def visualise_predictions(images, masks_gt, masks_pred, prostate_label,
                          save_path, titles=None):
    """Four-panel comparison per sample; difference panel is prostate-only."""
    n = len(images)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n), squeeze=False)
    column_titles = ["Input MRI", "Ground Truth", "Prediction", "Prostate error"]

    for i in range(n):
        img, gt, pred = images[i], masks_gt[i], masks_pred[i]

        axes[i, 0].imshow(img, cmap="gray", aspect="auto")
        for col, m in ((1, gt), (2, pred)):
            axes[i, col].imshow(img, cmap="gray", aspect="auto")
            axes[i, col].imshow(m, cmap=SEG_COLORS, vmin=0, vmax=NUM_CLASSES - 1,
                                aspect="auto", alpha=0.5)

        g = gt == prostate_label
        p = pred == prostate_label
        diff = np.zeros((*gt.shape, 4), dtype=np.float32)
        diff[g & p] = [0, 0.8, 0, 0.8]       # true positive
        diff[~g & p] = [0.9, 0, 0, 0.8]      # false positive (over-contour)
        diff[g & ~p] = [0, 0.3, 1.0, 0.8]    # false negative (under-contour)
        axes[i, 3].imshow(img, cmap="gray", aspect="auto")
        axes[i, 3].imshow(diff, aspect="auto")

        for col in range(4):
            axes[i, col].axis("off")
            if i == 0:
                axes[i, col].set_title(column_titles[col])
        if titles is not None:
            axes[i, 0].set_ylabel(titles[i])
            axes[i, 0].text(2, 10, titles[i], color="yellow", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


@torch.no_grad()
def run_inference(model, dataset, device):
    """Run inference over the dataset.

    Returns images, gt masks, predicted masks, global per-class Dice,
    and mean per-slice latency in ms (batch size 1, GPU-synchronised).
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                         num_workers=2, pin_memory=True)
    inter = torch.zeros(NUM_CLASSES, device=device)
    pred_sum = torch.zeros(NUM_CLASSES, device=device)
    target_sum = torch.zeros(NUM_CLASSES, device=device)
    all_images, all_gt, all_pred, latencies = [], [], [], []

    for k, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if k >= 5:  # skip warm-up iterations
            latencies.append((time.perf_counter() - t0) * 1000)

        i, p, t = dice_counts(logits, masks, NUM_CLASSES)
        inter += i
        pred_sum += p
        target_sum += t

        all_images.append(images[:, 0].cpu().numpy())
        all_gt.append(masks.cpu().numpy())
        all_pred.append(logits.argmax(dim=1).cpu().numpy())

    dice = counts_to_dice(inter, pred_sum, target_sum).tolist()
    return (np.concatenate(all_images), np.concatenate(all_gt),
            np.concatenate(all_pred), dice, float(np.mean(latencies)))


def main():
    parser = argparse.ArgumentParser(description="Inference with trained 2D U-Net")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="./predictions")
    parser.add_argument("--num_vis", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt = load_model(args.checkpoint, device)
    prostate_label = ckpt.get("prostate_label", 5)

    test_dataset = HipMRIDataset(
        os.path.join(args.data_root, "keras_slices_test"),
        os.path.join(args.data_root, "keras_slices_seg_test"),
    )
    print(f"Test set size: {len(test_dataset)}")

    images, gt, pred, dice, latency_ms = run_inference(model, test_dataset, device)

    print("\n--- Test Dice (global, accumulated over all slices) ---")
    for c, d in enumerate(dice):
        print(f"  {c} {CLASS_NAMES.get(c, '?'):<10} {d:.4f}")
    print(f"Mean inference latency: {latency_ms:.2f} ms/slice")

    # Per-slice prostate Dice, prostate-containing slices only
    g = gt == prostate_label
    p = pred == prostate_label
    inter = (g & p).sum(axis=(1, 2))
    denom = g.sum(axis=(1, 2)) + p.sum(axis=(1, 2))
    slice_dice = np.where(denom > 0, 2 * inter / np.maximum(denom, 1), np.nan)
    has_prostate = np.where(g.any(axis=(1, 2)))[0]
    fp_slices = np.where(~g.any(axis=(1, 2)) & p.any(axis=(1, 2)))[0]
    print(f"Per-slice prostate Dice (prostate slices): "
          f"{np.nanmean(slice_dice[has_prostate]):.4f}")
    print(f"Prostate predicted on {len(fp_slices)} prostate-free slices")

    def label(idx):
        meta = test_dataset.get_meta(idx)
        return (f"case {meta['case']} wk {meta['week']} sl {meta['slice']} "
                f"| Dice {slice_dice[idx]:.2f}")

    # Spread across the Dice distribution: worst → best
    order = has_prostate[np.argsort(slice_dice[has_prostate])]
    pick = order[np.linspace(0, len(order) - 1, args.num_vis, dtype=int)]
    visualise_predictions(images[pick], gt[pick], pred[pick], prostate_label,
                          os.path.join(args.output_dir, "prediction_comparison.png"),
                          titles=[label(i) for i in pick])

    worst = order[:3]
    visualise_predictions(images[worst], gt[worst], pred[worst], prostate_label,
                          os.path.join(args.output_dir, "worst_cases.png"),
                          titles=[label(i) for i in worst])

    if len(fp_slices) > 0:
        fp_pick = fp_slices[:3]
        visualise_predictions(images[fp_pick], gt[fp_pick], pred[fp_pick],
                              prostate_label,
                              os.path.join(args.output_dir, "false_positive_slices.png"),
                              titles=[label(i).split("|")[0] for i in fp_pick])


if __name__ == "__main__":
    main()
