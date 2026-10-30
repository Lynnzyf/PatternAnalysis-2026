"""
verify_2d_3d_alignment.py — second audit: 2D<->3D correspondence, slice orientation, 3D labels.

Usage:
    python verify_2d_3d_alignment.py <semantic_MRs> <semantic_labels_only> <keras_slices_data> [out.png]

1. Correlates 2D test slices of case 40 (week 0) with every slice of the 3D
   volume of patient 040 along the axis whose slice shape matches (axis 1),
   and with patient 042 as a control. corr ~1.0 at one index => same patient,
   and that index is the 2D slice's position in the volume.
2. Saves a figure: 3D axial slice through the prostate with labels 4 (cyan)
   and 5 (magenta), the matched 3D slice, and the 2D slice side by side.
"""

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

mr_dir, lab_dir, slices_root = sys.argv[1:4]
out_png = sys.argv[4] if len(sys.argv) > 4 else "verify_2d_3d_alignment.png"


_VOL_RE = re.compile(r"[A-Za-z]+(\d+)_week(\d+)", re.IGNORECASE)


def vol_path(directory, number, week, tag=None):
    """Find a volume by parsed patient number and week (prefix letter and suffix ignored)."""
    directory = os.path.expanduser(directory)
    if not os.path.isdir(directory):
        sys.exit(f"not a directory: {directory!r}")
    names = sorted(n for n in os.listdir(directory) if n.endswith((".nii", ".nii.gz")))
    for n in names:
        m = _VOL_RE.match(n)
        if m and int(m.group(1)) == number and int(m.group(2)) == week:
            return os.path.join(directory, n)
    near = [n for n in names if f"{number:03d}" in n][:5]
    sys.exit(f"no volume for patient {number:03d} week {week} in {directory!r}\n"
             f"  {len(names)} NIfTI files there; names containing '{number:03d}': {near}\n"
             f"  first files: {names[:3]}")


def corr(a, b):
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float((a * b).mean())


vol40 = nib.load(vol_path(mr_dir, 40, 0, "LFOV")).get_fdata()
vol42 = nib.load(vol_path(mr_dir, 42, 0, "LFOV")).get_fdata()
seg40 = nib.load(vol_path(lab_dir, 40, 0, "SEMANTIC")).get_fdata().astype(int)
print("3D volume shape:", vol40.shape)

matches = {}
for k in (16, 24, 31):
    f = os.path.join(slices_root, "keras_slices_test", f"case_040_week_0_slice_{k}.nii.gz")
    if not os.path.exists(f):
        print("missing", f)
        continue
    s2d = nib.load(f).get_fdata()
    best = {}
    for name, vol in (("040", vol40), ("042", vol42)):
        if vol.shape[0] != s2d.shape[0] or vol.shape[2] != s2d.shape[1]:
            print(f"  shape mismatch: 2D {s2d.shape} vs 3D {vol.shape}")
            continue
        scores = [corr(s2d, vol[:, j, :]) for j in range(vol.shape[1])]
        j = int(np.argmax(scores))
        best[name] = (j, round(scores[j], 4))
    matches[k] = best.get("040", (None, None))[0]
    print(f"2D case 40 slice {k}: best axis-1 index / corr -> "
          f"patient 040: {best.get('040')}, patient 042 (control): {best.get('042')}")

# Figure ---------------------------------------------------------------------
z = int(np.nonzero(seg40 == 5)[2].mean())  # axial (axis 2) index through label 5
k = 24 if matches.get(24) is not None else next((kk for kk, j in matches.items() if j is not None), None)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(vol40[:, :, z], cmap="gray")
overlay = np.ma.masked_where(seg40[:, :, z] < 4, seg40[:, :, z])
ax[0].imshow(overlay, cmap="cool", vmin=4, vmax=5, alpha=0.6)
ax[0].set_title(f"3D patient 040, axis-2 slice {z}\nlabel 4 = cyan, label 5 = magenta")

if k is not None and matches[k] is not None:
    j = matches[k]
    ax[1].imshow(vol40[:, j, :], cmap="gray", aspect="auto")
    ax[1].set_title(f"3D patient 040, axis-1 slice {j}")
    s2d = nib.load(os.path.join(slices_root, "keras_slices_test",
                                f"case_040_week_0_slice_{k}.nii.gz")).get_fdata()
    ax[2].imshow(s2d, cmap="gray", aspect="auto")
    ax[2].set_title(f"2D case 40 slice {k}")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig(out_png, dpi=110)
print("saved", out_png)
