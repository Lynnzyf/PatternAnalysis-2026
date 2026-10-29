"""
audit_3d.py — one-off audit of the HipMRI 3D volumes before fixing the 3D pipeline.

Usage:
    python audit_3d.py <semantic_MRs dir> <semantic_labels_only dir> <keras_slices_data dir>

Reports:
  1. volume count, patients, weeks per patient
  2. volume shapes and voxel spacing
  3. label values and voxel share (which label is the prostate in 3D?)
  4. whether 2D case IDs and 3D patient IDs refer to the same patients
     (pixel correlation between a 2D slice and slices of 3D volumes)
"""

import collections
import glob
import os
import re
import sys

import nibabel as nib
import numpy as np

mr_dir, lab_dir, slices_root = sys.argv[1], sys.argv[2], sys.argv[3]

# ---------------------------------------------------------------- 1-3. volumes
mrs = sorted(glob.glob(os.path.join(mr_dir, "*.nii.gz")))
print(f"{len(mrs)} MR volumes")

shapes, zooms, label_vox, label_vols = (collections.Counter() for _ in range(4))
weeks = collections.defaultdict(list)
vol_index = {}  # (patient number, week) -> path
missing = []

for p in mrs:
    name = os.path.basename(p)
    m = re.match(r"([A-Za-z]+)(\d+)_Week(\d+)", name)
    if m is None:
        print("  unparsed name:", name)
        continue
    num, wk = int(m.group(2)), int(m.group(3))
    weeks[m.group(1) + m.group(2)].append(wk)
    vol_index[(num, wk)] = p

    img = nib.load(p)
    shapes[img.shape] += 1
    zooms[tuple(round(float(z), 2) for z in img.header.get_zooms())] += 1

    seg_candidates = glob.glob(os.path.join(lab_dir, name.split("_LFOV")[0] + "*"))
    if not seg_candidates:
        missing.append(name)
        continue
    seg = np.asarray(nib.load(seg_candidates[0]).dataobj).astype(int)
    u, c = np.unique(seg, return_counts=True)
    label_vox.update(dict(zip(u.tolist(), c.tolist())))
    label_vols.update(u.tolist())

print(f"{len(weeks)} patients; weeks per patient: "
      f"{collections.Counter(len(v) for v in weeks.values())}")
print("patient IDs:", sorted(weeks))
print("shapes:", dict(shapes))
print("voxel spacing (mm):", dict(zooms))
total = sum(label_vox.values())
print("label voxel share:", {k: f"{v / total:.4f}" for k, v in sorted(label_vox.items())})
print("volumes containing label:", dict(sorted(label_vols.items())))
print("MRs without a label file:", missing[:5], f"({len(missing)} total)")

# ------------------------------------------------- 4. 2D <-> 3D patient mapping
def corr(a, b):
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float((a * b).mean())

def resize_nearest(x, shape):
    r = np.linspace(0, x.shape[0] - 1, shape[0]).round().astype(int)
    c = np.linspace(0, x.shape[1] - 1, shape[1]).round().astype(int)
    return x[np.ix_(r, c)]

split_cases = {}
for split in ["train", "validate", "test"]:
    files = glob.glob(os.path.join(slices_root, f"keras_slices_{split}", "*.nii.gz"))
    split_cases[split] = sorted({int(re.search(r"case_(\d+)", f).group(1)) for f in files})
    print(f"2D {split} case numbers:", split_cases[split])

nums_3d = {n for n, _ in vol_index}
for split, cases in split_cases.items():
    print(f"2D {split} cases with a 3D volume of the same number: "
          f"{len(set(cases) & nums_3d)}/{len(cases)}")

print("\nPixel check (does 2D case N == 3D patient N?)")
test_files = sorted(glob.glob(os.path.join(slices_root, "keras_slices_test",
                                           "case_*_week_0_slice_*.nii.gz")))
mid = [f for f in test_files if re.search(r"slice_(\d+)", f).group(1) in {"20", "30"}][:3]
for f in mid:
    case = int(re.search(r"case_(\d+)", f).group(1))
    k = int(re.search(r"slice_(\d+)", f).group(1))
    s2d = nib.load(f).get_fdata()
    scores = []
    for (num, wk), p in vol_index.items():
        if wk != 0:
            continue
        vol = nib.load(p)
        for axis in range(3):
            if k >= vol.shape[axis]:
                continue
            idx = [slice(None)] * 3
            idx[axis] = k
            s3d = np.asarray(vol.dataobj[tuple(idx)], dtype=np.float32)
            scores.append((corr(s2d, resize_nearest(s3d, s2d.shape)), num, axis))
    scores.sort(reverse=True)
    print(f"  2D case {case} slice {k} (shape {s2d.shape}): best matches "
          f"{[(round(c, 3), n, ax) for c, n, ax in scores[:3]]}")
