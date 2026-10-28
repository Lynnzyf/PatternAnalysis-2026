"""
compare_2d_3d.py — 2D U-Net baseline vs 3D Improved U-Net on the SAME 3D test volumes

Why: the 2D slices are coronal slices of the 3D volumes (axis 1, verified by
correlation with patient 040), so the 2D model can be run directly on every
coronal slice of a 3D test volume. Its predictions are stacked back into a
volume and scored with exactly the same evaluator as the 3D model
(predict.edge_analysis): native resolution, base/apex on axial slices,
false-positive slices. Both use the same test patients (40-42), which the
2D model never saw in training.

Inputs:
  --ckpt2d   best_model.pth of the 2D baseline
  --code2d   folder containing the 2D modules.py
  --preds3d  'preds' folder written by predict.py (one .npz per volume)

Usage:
    python compare_2d_3d.py --image_dir .../semantic_MRs --mask_dir .../semantic_labels_only \
        --ckpt2d .../baseline/best_model.pth --code2d .../HipMRI_UNet_s4983493 \
        --preds3d .../predictions3d/preds --save_dir .../comparison
"""

import argparse
import importlib.util
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataset import get_splits, load_volume
from predict import edge_analysis, error_overlay

SHAPE_2D = (256, 128)


def keep_largest(pred: np.ndarray, label: int) -> np.ndarray:
    """Post-processing: keep only the largest 3D connected component of `label`.

    Removed voxels become background. Used to test how much of each model's
    error is isolated false-positive specks.
    """
    from scipy import ndimage
    m = pred == label
    comp, n = ndimage.label(m)
    if n <= 1:
        return pred
    sizes = ndimage.sum(m, comp, range(1, n + 1))
    out = pred.copy()
    out[m & (comp != int(np.argmax(sizes)) + 1)] = 0
    return out


def load_2d_model(code_dir, ckpt_path, device):
    spec = importlib.util.spec_from_file_location("modules2d", os.path.join(code_dir, "modules.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = mod.UNet(in_channels=1, num_classes=6,
                     base_filters=ckpt.get("base_filters", 64)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("prostate_label", 5)


@torch.no_grad()
def predict_2d_volume(model, vol_zxy, device, chunk=32):
    """Run the 2D model on every coronal slice. vol_zxy: (Z, X, Y) -> pred (Z, X, Y)."""
    vol = torch.from_numpy(vol_zxy).to(device)
    slices = vol.permute(2, 1, 0).unsqueeze(1)            # (Y, 1, X, Z) = 2D layout
    native = slices.shape[-2:]
    if tuple(native) != SHAPE_2D:
        slices = F.interpolate(slices, size=SHAPE_2D, mode="bilinear", align_corners=False)
    mean = slices.mean(dim=(1, 2, 3), keepdim=True)
    std = slices.std(dim=(1, 2, 3), keepdim=True)
    slices = torch.where(std > 0, (slices - mean) / std.clamp(min=1e-6), slices)
    preds = []
    for i in range(0, slices.shape[0], chunk):
        logits = model(slices[i:i + chunk])
        if tuple(native) != SHAPE_2D:
            logits = F.interpolate(logits, size=native, mode="bilinear", align_corners=False)
        preds.append(logits.argmax(dim=1))
    pred = torch.cat(preds)                                # (Y, X, Z)
    return pred.permute(2, 1, 0).cpu().numpy().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--ckpt2d", required=True)
    ap.add_argument("--code2d", required=True)
    ap.add_argument("--preds3d", required=True)
    ap.add_argument("--save_dir", default="./comparison")
    ap.add_argument("--edge_k", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model2d, label = load_2d_model(args.code2d, args.ckpt2d, device)

    rows, worst, lat = [], [], []
    for n, (img_path, mask_path, meta) in enumerate(get_splits(args.image_dir, args.mask_dir)["test"]):
        vol = load_volume(img_path)
        gt = load_volume(mask_path).round().astype(np.uint8)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred2d = predict_2d_volume(model2d, vol, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if n > 0:
            lat.append((time.perf_counter() - t0) * 1000)
        pred3d = np.load(os.path.join(args.preds3d, meta["name"] + ".npz"))["pred"]

        r2 = edge_analysis(pred2d, gt, label, k=args.edge_k)
        r3 = edge_analysis(pred3d, gt, label, k=args.edge_k)
        r2l = edge_analysis(keep_largest(pred2d, label), gt, label, k=args.edge_k)
        r3l = edge_analysis(keep_largest(pred3d, label), gt, label, k=args.edge_k)
        rows.append((meta, r2, r3, r2l, r3l))
        for which in ("base", "apex"):
            for z in r2[which]:
                worst.append((r2["slice_dice"][z] - r3["slice_dice"][z], r2["slice_dice"][z],
                              f"{meta['name']} {which} z={z} | 2D {r2['slice_dice'][z]:.2f} "
                              f"3D {r3['slice_dice'][z]:.2f}",
                              vol[z], gt[z], pred2d[z], pred3d[z]))
        print(f"{meta['name']:<14} volume Dice 2D {r2['volume_dice']:.3f} / 3D {r3['volume_dice']:.3f} | "
              f"base {r2['base_dice']:.2f}/{r3['base_dice']:.2f} | apex {r2['apex_dice']:.2f}/"
              f"{r3['apex_dice']:.2f} | FP slices {r2['fp_slices']}/{r3['fp_slices']}")

    def summary(idx, name):
        rs = [r[idx] for r in rows]
        m = lambda k: np.nanmean([r[k] for r in rs])
        s = lambda k: int(sum(r[k] for r in rs))
        n_edge = len(rs) * args.edge_k
        return (f"{name}: volume Dice {m('volume_dice'):.4f} | base {m('base_dice'):.4f} "
                f"(missed {s('base_missed')}/{n_edge}) | apex {m('apex_dice'):.4f} "
                f"(missed {s('apex_missed')}/{n_edge}) | inner {m('inner_dice'):.4f} | "
                f"FP slices {s('fp_slices')}")

    patients = sorted({r[0]["patient"] for r in rows})
    per_patient, patient_means = [], []
    for p in patients:
        a = np.array([(r[1]["volume_dice"], r[2]["volume_dice"]) for r in rows
                      if r[0]["patient"] == p])
        patient_means.append(a.mean(axis=0))
        per_patient.append(f"  {p}: {len(a)} volumes, 2D {a[:, 0].mean():.4f}, 3D {a[:, 1].mean():.4f}")
    pm = np.array(patient_means).mean(axis=0)
    per_patient.append(f"  mean of patient means: 2D {pm[0]:.4f}, 3D {pm[1]:.4f}")

    lines = [f"2D vs 3D on {len(rows)} test volumes (native resolution, edge k={args.edge_k})",
             summary(1, "2D U-Net "), summary(2, "3D Imp. U-Net"),
             "After keeping the largest connected prostate component:",
             summary(3, "2D U-Net  + LCC"), summary(4, "3D Imp. U-Net + LCC"),
             "Per patient (mean volume Dice, raw):", *per_patient,
             f"2D latency: {np.mean(lat):.1f} ms/volume (all coronal slices, fp32)"]
    print("\n" + "\n".join(lines))
    with open(os.path.join(args.save_dir, "comparison.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # Edge slices where 2D is much worse than 3D (and the reverse) ------------
    worst.sort(key=lambda r: r[0])           # (2D Dice - 3D Dice), ascending
    picks = worst[:3] + worst[-1:]           # 3 biggest 3D wins, then the biggest 2D win
    fig, axes = plt.subplots(len(picks), 4, figsize=(14, 3.6 * len(picks)), squeeze=False)
    for i, (_, _, title, img, gt, p2, p3) in enumerate(picks):
        axes[i, 0].imshow(img, cmap="gray", aspect="auto")
        axes[i, 0].text(2, 10, title, color="yellow", fontsize=8)
        axes[i, 1].imshow(img, cmap="gray", aspect="auto")
        axes[i, 1].imshow(np.ma.masked_where(gt != label, gt), cmap="autumn", alpha=0.6, aspect="auto")
        for col, p in ((2, p2), (3, p3)):
            axes[i, col].imshow(img, cmap="gray", aspect="auto")
            axes[i, col].imshow(error_overlay(gt, p, label), aspect="auto")
        for col, t in enumerate(["Input (axial)", "GT prostate", "2D U-Net error", "3D U-Net error"]):
            axes[i, col].axis("off")
            if i == 0:
                axes[i, col].set_title(t)
    plt.tight_layout()
    out = os.path.join(args.save_dir, "edge_slices_2d_vs_3d.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved", out)


if __name__ == "__main__":
    main()
