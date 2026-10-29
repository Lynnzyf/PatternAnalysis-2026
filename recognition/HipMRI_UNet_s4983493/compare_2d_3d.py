"""
compare_2d_3d.py — 2D U-Net vs 3D Improved U-Net on the SAME 3D test volumes

The 2D slices are coronal slices of the 3D volumes (verified by image
correlation for patient 040), so the 2D model is run on every coronal slice
of each 3D test volume, its predictions are stacked back into a volume, and
both models are scored by the same evaluator (predict.edge_analysis) at
native resolution: volume Dice, base/apex axial slices, false-positive slices.
The test patients (40-42) are unseen by both models.

Reported variants:
  - raw predictions of both models
  - + largest-connected-component post-processing (predict.keep_largest)
  - 2D restricted to the coronal slab the 2D dataset covers (60 slices per
    volume, starting at index 97); the script checks that all GT prostate
    lies inside this slab
Plus a profile of false-positive voxels along the coronal axis, latency of the
2D model per volume (all 256 coronal slices and the 60-slice slab), and a
figure of the edge slices with the largest 2D/3D differences.

Inputs:
  --ckpt2d   2D checkpoint from train.py --model 2d
  --preds3d  'preds' folder written by predict.py --model 3d

Usage:
    python compare_2d_3d.py --image_dir .../semantic_MRs --mask_dir .../semantic_labels_only \
        --ckpt2d .../checkpoints_2d_seed1/best_model.pth \
        --preds3d .../predictions_3d_seed1/preds --save_dir .../comparison_seed1
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataset import PROSTATE_LABEL, SHAPE_2D, get_splits_3d, load_volume_3d
from predict import edge_analysis, error_overlay, keep_largest, load_checkpoint, sync


@torch.no_grad()
def predict_2d_volume(model, vol_zxy, device, y_range=None, chunk=32):
    """Run the 2D model on the coronal slices of a (Z, X, Y) volume -> pred (Z, X, Y).

    y_range: optional slice of coronal indices to run on (others stay background).
    """
    vol = torch.from_numpy(vol_zxy).to(device)
    slices = vol.permute(2, 1, 0).unsqueeze(1)            # (Y, 1, X, Z) = 2D layout
    ys = torch.arange(slices.shape[0], device=device)
    if y_range is not None:
        ys, slices = ys[y_range], slices[y_range]
    native = slices.shape[-2:]
    if tuple(native) != SHAPE_2D:
        slices = F.interpolate(slices, size=SHAPE_2D, mode="bilinear", align_corners=False)
    mean = slices.mean(dim=(1, 2, 3), keepdim=True)
    std = slices.std(dim=(1, 2, 3), keepdim=True)
    slices = torch.where(std > 0, (slices - mean) / std.clamp(min=1e-6), slices)
    out = torch.zeros((vol.shape[2],) + tuple(native), dtype=torch.long, device=device)
    for i in range(0, slices.shape[0], chunk):
        logits = model(slices[i:i + chunk])
        if tuple(native) != SHAPE_2D:
            logits = F.interpolate(logits, size=native, mode="bilinear", align_corners=False)
        out[ys[i:i + chunk]] = logits.argmax(dim=1)
    return out.permute(2, 1, 0).cpu().numpy().astype(np.uint8)   # (Y, X, Z) -> (Z, X, Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--ckpt2d", required=True)
    ap.add_argument("--preds3d", required=True)
    ap.add_argument("--save_dir", default="./comparison")
    ap.add_argument("--edge_k", type=int, default=2)
    ap.add_argument("--band_start", type=int, default=97,
                    help="First coronal (Y) index of the slab covered by the 2D dataset "
                         "(97-98 measured for patient 040)")
    ap.add_argument("--band_len", type=int, default=60,
                    help="Coronal slices per volume in the 2D dataset (60 for every split)")
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model2d, _ = load_checkpoint("2d", args.ckpt2d, device)
    label = PROSTATE_LABEL

    rows, worst, lat, lat_band, gt_in_band = [], [], [], [], []
    band = slice(args.band_start, args.band_start + args.band_len)
    fp2_prof = fp3_prof = gt_prof = None
    for n, (img_path, mask_path, meta) in enumerate(get_splits_3d(args.image_dir, args.mask_dir)["test"]):
        vol = load_volume_3d(img_path)
        gt = load_volume_3d(mask_path).round().astype(np.uint8)
        sync(device)
        t0 = time.perf_counter()
        pred2d = predict_2d_volume(model2d, vol, device)
        sync(device)
        t1 = time.perf_counter()
        predict_2d_volume(model2d, vol, device, y_range=band)   # timing of slab-only inference
        sync(device)
        if n > 0:  # first volume is warm-up
            lat.append((t1 - t0) * 1000)
            lat_band.append((time.perf_counter() - t1) * 1000)
        pred3d = np.load(os.path.join(args.preds3d, meta["name"] + ".npz"))["pred"]

        r2 = edge_analysis(pred2d, gt, label, k=args.edge_k)
        r3 = edge_analysis(pred3d, gt, label, k=args.edge_k)
        r2l = edge_analysis(keep_largest(pred2d, label), gt, label, k=args.edge_k)
        r3l = edge_analysis(keep_largest(pred3d, label), gt, label, k=args.edge_k)

        # 2D restricted to the coronal slab the 2D dataset covers
        pred2d_band = np.zeros_like(pred2d)
        pred2d_band[:, :, band] = pred2d[:, :, band]
        r2b = edge_analysis(pred2d_band, gt, label, k=args.edge_k)
        g = gt == label
        gt_in_band.append(g[:, :, band].sum() / max(g.sum(), 1))

        # false-positive prostate voxels per coronal index (axis 2 = Y)
        fp2 = ((pred2d == label) & ~g).sum(axis=(0, 1))
        fp3 = ((pred3d == label) & ~g).sum(axis=(0, 1))
        gp = g.sum(axis=(0, 1))
        if fp2_prof is None or len(fp2_prof) != len(fp2):
            fp2_prof, fp3_prof, gt_prof = fp2.astype(float), fp3.astype(float), gp.astype(float)
        else:
            fp2_prof += fp2
            fp3_prof += fp3
            gt_prof += gp

        rows.append((meta, r2, r3, r2l, r3l, r2b))
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
             f"2D restricted to the 2D-dataset coronal slab [{band.start}, {band.stop}) "
             f"(GT prostate inside slab: min {min(gt_in_band):.3f} over volumes):",
             summary(5, "2D U-Net + slab"),
             "Per patient (mean volume Dice, raw):", *per_patient,
             f"2D latency: {np.mean(lat):.1f} ms/volume (all coronal slices), "
             f"{np.mean(lat_band):.1f} ms/volume ({args.band_len}-slice slab only), fp32, "
             f"chunks of 32 slices"]
    print("\n" + "\n".join(lines))
    with open(os.path.join(args.save_dir, "comparison.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # False positives along the coronal axis ---------------------------------
    y = np.arange(len(fp2_prof))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axvspan(band.start, band.stop, color="grey", alpha=0.15, label="2D training slab")
    ax.plot(y, fp2_prof + 1, label="2D U-Net false positives")
    ax.plot(y, fp3_prof + 1, label="3D U-Net false positives")
    ax.plot(y, gt_prof + 1, "k--", linewidth=1, label="GT prostate voxels")
    ax.set_yscale("log")
    ax.set_xlabel("Coronal slice index (anterior-posterior axis)")
    ax.set_ylabel("Voxels + 1 (summed over test volumes)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(args.save_dir, "fp_profile_coronal.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved", out)

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
