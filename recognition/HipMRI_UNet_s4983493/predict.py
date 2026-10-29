"""
predict.py — inference, evaluation and visualisation for a trained model

    --model 2d : evaluates the 2D U-Net on the 2D test slices
                 (global / per-slice prostate Dice, false-positive slices,
                 latency per slice, figures of best-to-worst and worst slices)
    --model 3d : evaluates the 3D Improved U-Net on the 3D test volumes at
                 native resolution (logits upsampled back if trained on a
                 coarser grid), with the base/apex analysis, latency per
                 volume, peak VRAM, figures, and one .npz prediction per
                 volume for compare_2d_3d.py

Shared helpers used by compare_2d_3d.py: edge_analysis, keep_largest,
error_overlay, plot_panels.

NumPy is allowed in this file (task specification).

Examples:
    python predict.py --model 2d --data_root .../keras_slices_data \
        --checkpoint .../checkpoints_2d_seed1/best_model.pth --save_dir .../predictions_2d
    python predict.py --model 3d --image_dir .../semantic_MRs --mask_dir .../semantic_labels_only \
        --checkpoint .../checkpoints_3d_seed1/best_model.pth --save_dir .../predictions_3d_seed1
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

from dataset import (NUM_CLASSES, PROSTATE_LABEL, BLADDER_LABEL, SHAPE_3D, HipMRI2DDataset,
                     get_splits_3d, load_volume_3d, preprocess_3d)
from modules import UNet2D, ImprovedUNet3D, dice_counts, counts_to_dice

SEG_COLORS = ListedColormap(["black", "#7f8c8d", "#f1c40f", "#3498db", "#9b59b6", "#e74c3c"])
CLASS_NAMES = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]


# ===========================================================================
# Shared analysis and plotting helpers
# ===========================================================================

def slice_dice(p: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Per-axial-slice Dice of boolean (D, H, W) arrays; NaN where both are empty."""
    inter = (p & g).sum(axis=(1, 2))
    denom = p.sum(axis=(1, 2)) + g.sum(axis=(1, 2))
    return np.where(denom > 0, 2 * inter / np.maximum(denom, 1), np.nan)


def edge_analysis(pred: np.ndarray, gt: np.ndarray, label: int = PROSTATE_LABEL,
                  bladder_label: int = BLADDER_LABEL, k: int = 2) -> dict:
    """Base/apex analysis of one (D, H, W) volume along the superior-inferior axis.

    The first/last k prostate-containing axial slices of the ground truth are
    the gland's ends; the end closer to the bladder is the base, the other the
    apex. Also counts false-positive slices (prostate predicted where the GT
    slice has none).
    """
    g, p = gt == label, pred == label
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


def keep_largest(pred: np.ndarray, label: int = PROSTATE_LABEL) -> np.ndarray:
    """Post-processing: keep only the largest 3D connected component of `label`."""
    from scipy import ndimage
    m = pred == label
    comp, n = ndimage.label(m)
    if n <= 1:
        return pred
    sizes = ndimage.sum(m, comp, range(1, n + 1))
    out = pred.copy()
    out[m & (comp != int(np.argmax(sizes)) + 1)] = 0
    return out


def error_overlay(gt2d, pred2d, label=PROSTATE_LABEL):
    """RGBA overlay: green TP, red FP (over-contour), blue FN (under-contour)."""
    g, p = gt2d == label, pred2d == label
    rgba = np.zeros((*g.shape, 4), dtype=np.float32)
    rgba[g & p] = [0, 0.8, 0, 0.8]
    rgba[~g & p] = [0.9, 0, 0, 0.8]
    rgba[g & ~p] = [0, 0.3, 1.0, 0.8]
    return rgba


def plot_panels(rows, save_path, label=PROSTATE_LABEL):
    """rows: list of (title, image2d, gt2d, pred2d) -> Input / GT / Prediction / error."""
    fig, axes = plt.subplots(len(rows), 4, figsize=(14, 3.6 * len(rows)), squeeze=False)
    titles = ["Input MRI", "Ground Truth", "Prediction", "Prostate error"]
    for i, (title, img, gt, pred) in enumerate(rows):
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


def load_checkpoint(model_type, path, device):
    """Load a checkpoint from train.py (also accepts the earlier 2D/3D formats)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    a = ckpt.get("args", {})
    if model_type == "2d":
        model = UNet2D(1, NUM_CLASSES, a.get("base_filters", ckpt.get("base_filters", 64)))
    else:
        model = ImprovedUNet3D(1, NUM_CLASSES, a.get("base_filters", 16), a.get("dropout", 0.3))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded {path} (epoch {ckpt.get('epoch', '?')}, "
          f"val prostate Dice {ckpt.get('best_dice', float('nan')):.4f})")
    return model, ckpt


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


# ===========================================================================
# 2D evaluation
# ===========================================================================

@torch.no_grad()
def predict_2d(args, device):
    model, _ = load_checkpoint("2d", args.checkpoint, device)
    label = PROSTATE_LABEL
    ds = HipMRI2DDataset(os.path.join(args.data_root, "keras_slices_test"),
                         os.path.join(args.data_root, "keras_slices_seg_test"))
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    inter = torch.zeros(NUM_CLASSES, device=device)
    pred_sum = torch.zeros(NUM_CLASSES, device=device)
    gt_sum = torch.zeros(NUM_CLASSES, device=device)
    images, gts, preds, lat = [], [], [], []
    for k, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        sync(device)
        t0 = time.perf_counter()
        logits = model(x)
        sync(device)
        if k >= 5:
            lat.append((time.perf_counter() - t0) * 1000)
        i, p, t = dice_counts(logits, y, NUM_CLASSES)
        inter, pred_sum, gt_sum = inter + i, pred_sum + p, gt_sum + t
        images.append(x[0, 0].cpu().numpy())
        gts.append(y[0].cpu().numpy())
        preds.append(logits.argmax(1)[0].cpu().numpy())
    images, gts, preds = np.stack(images), np.stack(gts), np.stack(preds)
    dice = counts_to_dice(inter, pred_sum, gt_sum).tolist()

    g, p = gts == label, preds == label
    denom = g.sum(axis=(1, 2)) + p.sum(axis=(1, 2))
    sd = np.where(denom > 0, 2 * (g & p).sum(axis=(1, 2)) / np.maximum(denom, 1), np.nan)
    has = np.nonzero(g.any(axis=(1, 2)))[0]
    fp = np.nonzero(~g.any(axis=(1, 2)) & p.any(axis=(1, 2)))[0]

    lines = ["2D U-Net — 2D test slices",
             *[f"  {c} {CLASS_NAMES[c]:<10} {dice[c]:.4f}" for c in range(NUM_CLASSES)],
             f"Prostate Dice (global, pooled over all slices): {dice[label]:.4f}",
             f"Prostate Dice (per-slice mean, prostate slices): {np.nanmean(sd[has]):.4f}",
             f"Prostate predicted on {len(fp)} of {len(ds) - len(has)} prostate-free slices",
             f"Latency: {np.mean(lat):.2f} ms/slice (fp32, batch 1)"]
    print("\n".join(lines))
    with open(os.path.join(args.save_dir, "test_results_2d.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    def title(idx):
        m = ds.get_meta(idx)
        return f"case {m['case']} wk {m['week']} sl {m['slice']} | Dice {sd[idx]:.2f}"

    order = has[np.argsort(sd[has])]
    pick = order[np.linspace(0, len(order) - 1, args.num_vis, dtype=int)]
    plot_panels([(title(i), images[i], gts[i], preds[i]) for i in pick],
                os.path.join(args.save_dir, "prediction_comparison_2d.png"))
    plot_panels([(title(i), images[i], gts[i], preds[i]) for i in order[:3]],
                os.path.join(args.save_dir, "worst_slices_2d.png"))
    if len(fp):
        plot_panels([(title(i).split("|")[0], images[i], gts[i], preds[i]) for i in fp[:3]],
                    os.path.join(args.save_dir, "false_positive_slices_2d.png"))


# ===========================================================================
# 3D evaluation
# ===========================================================================

@torch.no_grad()
def predict_3d(args, device):
    model, ckpt = load_checkpoint("3d", args.checkpoint, device)
    label = PROSTATE_LABEL
    shape = tuple(ckpt.get("args", {}).get("target_shape", SHAPE_3D))
    os.makedirs(os.path.join(args.save_dir, "preds"), exist_ok=True)

    inter, pred_sum, gt_sum = (np.zeros(NUM_CLASSES) for _ in range(3))
    results, lat, edge_rows = [], [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for n, (img_path, mask_path, meta) in enumerate(
            get_splits_3d(args.image_dir, args.mask_dir)["test"]):
        x, _ = preprocess_3d(img_path, mask_path, shape)
        gt = load_volume_3d(mask_path).round().astype(np.uint8)
        x = x.float()[None, None].to(device)
        sync(device)
        t0 = time.perf_counter()
        logits = model(x)
        if tuple(logits.shape[2:]) != gt.shape:
            logits = F.interpolate(logits, size=gt.shape, mode="trilinear", align_corners=False)
        pred_t = logits.argmax(dim=1)[0]
        sync(device)
        if n > 0:  # first volume is warm-up
            lat.append((time.perf_counter() - t0) * 1000)
        pred = pred_t.cpu().numpy().astype(np.uint8)

        for c in range(NUM_CLASSES):
            p, g = pred == c, gt == c
            inter[c] += (p & g).sum()
            pred_sum[c] += p.sum()
            gt_sum[c] += g.sum()

        r = edge_analysis(pred, gt, label, k=args.edge_k)
        results.append(r)
        np.savez_compressed(os.path.join(args.save_dir, "preds", meta["name"] + ".npz"), pred=pred)

        image = load_volume_3d(img_path)
        for which in ("base", "apex"):
            for z in r[which]:
                edge_rows.append((r["slice_dice"][z],
                                  f"{meta['name']} {which} z={z} | Dice {r['slice_dice'][z]:.2f}",
                                  image[z], gt[z], pred[z]))
        if n == 0:
            zc = int(np.nonzero(gt == label)[0].mean())
            yc = int(np.nonzero(gt == label)[2].mean())
            # coronal plane shown as (X, Z), the same layout as the 2D slices
            plot_panels([(f"{meta['name']} axial z={zc}", image[zc], gt[zc], pred[zc]),
                         (f"{meta['name']} coronal y={yc}", image[:, :, yc].T,
                          gt[:, :, yc].T, pred[:, :, yc].T)],
                        os.path.join(args.save_dir, "overview_3d.png"))
        print(f"{meta['name']:<14} prostate {r['volume_dice']:.4f} | base {r['base_dice']:.3f} "
              f"({r['base_missed']} missed) | apex {r['apex_dice']:.3f} ({r['apex_missed']} missed) "
              f"| inner {r['inner_dice']:.3f} | FP slices {r['fp_slices']}")

    peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0
    dice = 2 * inter / np.maximum(pred_sum + gt_sum, 1)
    vd = np.array([r["volume_dice"] for r in results])
    mean = lambda key: np.nanmean([r[key] for r in results])
    tot = lambda key: int(sum(r[key] for r in results))
    n_edge = len(results) * args.edge_k
    lines = [
        f"3D Improved U-Net — native-resolution test results ({len(results)} volumes, grid {shape})",
        *[f"  {c} {CLASS_NAMES[c]:<10} {dice[c]:.4f}" for c in range(NUM_CLASSES)],
        f"Prostate Dice (global): {dice[label]:.4f}",
        f"Prostate Dice per volume: mean {vd.mean():.4f}, std {vd.std():.4f}, min {vd.min():.4f}",
        f"Base slices (k={args.edge_k}): mean Dice {mean('base_dice'):.4f}, "
        f"missed {tot('base_missed')}/{n_edge}",
        f"Apex slices (k={args.edge_k}): mean Dice {mean('apex_dice'):.4f}, "
        f"missed {tot('apex_missed')}/{n_edge}",
        f"Inner slices: mean Dice {mean('inner_dice'):.4f}",
        f"False-positive slices (total): {tot('fp_slices')}",
        f"Latency: {np.mean(lat):.1f} ms/volume (fp32, batch 1)",
        f"Peak inference VRAM: {peak:.2f} GB",
    ]
    print("\n" + "\n".join(lines))
    with open(os.path.join(args.save_dir, "test_results_3d_native.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    edge_rows.sort(key=lambda r: r[0])
    plot_panels([r[1:] for r in edge_rows[:4]],
                os.path.join(args.save_dir, "worst_base_apex_3d.png"))


def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained 2D or 3D model")
    ap.add_argument("--model", choices=["2d", "3d"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", help="2D: keras_slices_data folder")
    ap.add_argument("--image_dir", help="3D: semantic_MRs folder")
    ap.add_argument("--mask_dir", help="3D: semantic_labels_only folder")
    ap.add_argument("--save_dir", default=None)
    ap.add_argument("--num_vis", type=int, default=5, help="2D: slices in the overview figure")
    ap.add_argument("--edge_k", type=int, default=2, help="3D: slices counted as base / apex")
    args = ap.parse_args()
    args.save_dir = args.save_dir or f"./predictions_{args.model}"
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "2d":
        if not args.data_root:
            ap.error("--model 2d needs --data_root")
        predict_2d(args, device)
    else:
        if not (args.image_dir and args.mask_dir):
            ap.error("--model 3d needs --image_dir and --mask_dir")
        predict_3d(args, device)


if __name__ == "__main__":
    main()
