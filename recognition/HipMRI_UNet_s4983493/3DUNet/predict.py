"""
predict.py — Native-resolution evaluation and visualisation for the 3D Improved U-Net

For every test volume:
  - runs the model on the resampled 128^3 input, upsamples the logits back to
    the native grid (trilinear) and takes the argmax, so Dice is measured at
    the original resolution (comparable with the 2D baseline)
  - per-class Dice, per-volume prostate Dice
  - base/apex analysis along the superior-inferior axis (D):
      * the first/last k prostate-containing axial slices of the ground truth
      * the end closer to the bladder (label 3) is the base, the other the apex
      * slice Dice and "missed" slices (prostate in GT, none predicted)
  - false-positive slices (prostate predicted where the GT slice has none)
  - latency per volume (fp32, batch 1, includes upsampling) and peak VRAM
  - saves predicted label volumes (.npz) for the 2D-vs-3D comparison script

Figures: worst base/apex slices with a TP/FP/FN overlay, and an axial +
coronal overview of one test volume.

Usage:
    python predict.py --image_dir .../semantic_MRs --mask_dir .../semantic_labels_only \
        --checkpoint .../best_model.pth --save_dir .../predictions3d
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import torch.nn.functional as F

from dataset import get_splits, load_volume, preprocess, NUM_CLASSES
from modules import ImprovedUNet3D

SEG_COLORS = ListedColormap(["black", "#7f8c8d", "#f1c40f", "#3498db", "#9b59b6", "#e74c3c"])
CLASS_NAMES = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]


# ---------------------------------------------------------------- analysis
def slice_dice(p: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Per-axial-slice Dice of boolean (D, H, W) arrays; NaN where both empty."""
    inter = (p & g).sum(axis=(1, 2))
    denom = p.sum(axis=(1, 2)) + g.sum(axis=(1, 2))
    return np.where(denom > 0, 2 * inter / np.maximum(denom, 1), np.nan)


def edge_analysis(pred: np.ndarray, gt: np.ndarray, prostate_label: int = 5,
                  bladder_label: int = 3, k: int = 2) -> dict:
    """Base/apex analysis of one volume, slices along axis 0 (superior-inferior).

    Shared by the 3D model and (later) the 2D baseline re-assembled into volumes.
    """
    g, p = gt == prostate_label, pred == prostate_label
    g_any, p_any = g.any(axis=(1, 2)), p.any(axis=(1, 2))
    present = np.nonzero(g_any)[0]
    z0, z1 = int(present.min()), int(present.max())
    bladder_z = np.nonzero(gt == bladder_label)[0].mean()
    low = list(range(z0, min(z0 + k, z1 + 1)))
    high = list(range(max(z1 - k + 1, z0), z1 + 1))
    base, apex = (low, high) if abs(z0 - bladder_z) < abs(z1 - bladder_z) else (high, low)
    d = slice_dice(p, g)
    inner = [z for z in present if z not in base and z not in apex]
    return {
        "volume_dice": float(2 * (p & g).sum() / max(p.sum() + g.sum(), 1)),
        "base_dice": float(np.mean(d[base])),
        "apex_dice": float(np.mean(d[apex])),
        "inner_dice": float(np.nanmean(d[inner])) if inner else float("nan"),
        "base_missed": int((~p_any[base]).sum()),
        "apex_missed": int((~p_any[apex]).sum()),
        "fp_slices": int((p_any & ~g_any).sum()),
        "base": base, "apex": apex, "slice_dice": d,
    }


# ---------------------------------------------------------------- figures
def error_overlay(gt2d, pred2d, label):
    g, p = gt2d == label, pred2d == label
    rgba = np.zeros((*g.shape, 4), dtype=np.float32)
    rgba[g & p] = [0, 0.8, 0, 0.8]       # true positive
    rgba[~g & p] = [0.9, 0, 0, 0.8]      # false positive (over-contour)
    rgba[g & ~p] = [0, 0.3, 1.0, 0.8]    # false negative (under-contour)
    return rgba


def plot_rows(rows, label, save_path, transpose=False):
    """rows: list of (title, image2d, gt2d, pred2d)."""
    fig, axes = plt.subplots(len(rows), 4, figsize=(14, 3.6 * len(rows)), squeeze=False)
    titles = ["Input MRI", "Ground Truth", "Prediction", "Prostate error"]
    for i, (title, img, gt, pred) in enumerate(rows):
        if transpose:
            img, gt, pred = img.T, gt.T, pred.T
        axes[i, 0].imshow(img, cmap="gray", aspect="auto")
        for col, m in ((1, gt), (2, pred)):
            axes[i, col].imshow(img, cmap="gray", aspect="auto")
            axes[i, col].imshow(m, cmap=SEG_COLORS, vmin=0, vmax=NUM_CLASSES - 1,
                                alpha=0.5, aspect="auto")
        axes[i, 3].imshow(img, cmap="gray", aspect="auto")
        axes[i, 3].imshow(error_overlay(gt, pred, label), aspect="auto")
        axes[i, 0].text(2, 10, title, color="yellow", fontsize=8)
        for col in range(4):
            axes[i, col].axis("off")
            if i == 0:
                axes[i, col].set_title(titles[col])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved", save_path)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Evaluate the 3D Improved U-Net at native resolution")
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--save_dir", default="./predictions3d")
    ap.add_argument("--edge_k", type=int, default=2, help="Slices counted as base / apex")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.save_dir, "preds"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    targs = ckpt["args"]
    label = targs["prostate_label"]
    shape = tuple(targs["target_shape"])
    model = ImprovedUNet3D(1, NUM_CLASSES, targs["base_filters"], targs["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {args.checkpoint} (epoch {ckpt['epoch']}, "
          f"val prostate Dice {ckpt['best_dice']:.4f})")

    test_pairs = get_splits(args.image_dir, args.mask_dir)["test"]
    inter = np.zeros(NUM_CLASSES)
    pred_sum = np.zeros(NUM_CLASSES)
    gt_sum = np.zeros(NUM_CLASSES)
    results, latencies, edge_rows = [], [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for n, (img_path, mask_path, meta) in enumerate(test_pairs):
            image_ds, _ = preprocess(img_path, mask_path, shape)
            gt = load_volume(mask_path).round().astype(np.uint8)
            x = image_ds.float()[None, None].to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = F.interpolate(model(x), size=gt.shape, mode="trilinear",
                                   align_corners=False)
            pred_t = logits.argmax(dim=1)[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            if n > 0:  # first volume = warm-up
                latencies.append((time.perf_counter() - t0) * 1000)
            pred = pred_t.cpu().numpy().astype(np.uint8)

            for c in range(NUM_CLASSES):
                p, g = pred == c, gt == c
                inter[c] += (p & g).sum()
                pred_sum[c] += p.sum()
                gt_sum[c] += g.sum()

            r = edge_analysis(pred, gt, label, k=args.edge_k)
            r["name"] = meta["name"]
            results.append(r)
            np.savez_compressed(os.path.join(args.save_dir, "preds", meta["name"] + ".npz"),
                                pred=pred)

            image = load_volume(img_path)
            for which in ("base", "apex"):
                for z in r[which]:
                    edge_rows.append((r["slice_dice"][z],
                                      f"{meta['name']} {which} z={z} | Dice {r['slice_dice'][z]:.2f}",
                                      image[z], gt[z], pred[z]))
            if n == 0:
                zc = int(np.nonzero(gt == label)[0].mean())
                yc = int(np.nonzero(gt == label)[2].mean())
                plot_rows([(f"{meta['name']} axial z={zc}", image[zc], gt[zc], pred[zc]),
                           # coronal plane shown as (X, Z), same layout as the 2D slices
                           (f"{meta['name']} coronal y={yc}", image[:, :, yc].T,
                            gt[:, :, yc].T, pred[:, :, yc].T)],
                          label, os.path.join(args.save_dir, "overview.png"))
            print(f"{meta['name']:<14} prostate {r['volume_dice']:.4f} | "
                  f"base {r['base_dice']:.3f} ({r['base_missed']} missed) | "
                  f"apex {r['apex_dice']:.3f} ({r['apex_missed']} missed) | "
                  f"inner {r['inner_dice']:.3f} | FP slices {r['fp_slices']}")

    peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0
    dice = 2 * inter / np.maximum(pred_sum + gt_sum, 1)
    vd = np.array([r["volume_dice"] for r in results])
    agg = lambda key: np.nanmean([r[key] for r in results])
    tot = lambda key: int(sum(r[key] for r in results))
    n_edge = len(results) * args.edge_k

    lines = [
        f"3D Improved U-Net — native-resolution test results ({len(results)} volumes)",
        *[f"  {c} {CLASS_NAMES[c]:<10} {dice[c]:.4f}" for c in range(NUM_CLASSES)],
        f"Prostate Dice (global): {dice[label]:.4f}",
        f"Prostate Dice per volume: mean {vd.mean():.4f}, std {vd.std():.4f}, min {vd.min():.4f}",
        f"Base slices (k={args.edge_k}): mean Dice {agg('base_dice'):.4f}, "
        f"missed {tot('base_missed')}/{n_edge}",
        f"Apex slices (k={args.edge_k}): mean Dice {agg('apex_dice'):.4f}, "
        f"missed {tot('apex_missed')}/{n_edge}",
        f"Inner slices: mean Dice {agg('inner_dice'):.4f}",
        f"False-positive slices (total): {tot('fp_slices')}",
        f"Latency: {np.mean(latencies):.1f} ms/volume (fp32, batch 1, incl. upsampling)",
        f"Peak inference VRAM: {peak:.2f} GB",
    ]
    print("\n" + "\n".join(lines))
    with open(os.path.join(args.save_dir, "test_results_native.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    edge_rows.sort(key=lambda r: r[0])
    plot_rows([r[1:] for r in edge_rows[:4]], label,
              os.path.join(args.save_dir, "worst_base_apex.png"))


if __name__ == "__main__":
    main()
