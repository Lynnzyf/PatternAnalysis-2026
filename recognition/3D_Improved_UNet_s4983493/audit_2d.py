"""
audit_2d.py — audit of the 2D slice dataset (labels and patient-level split)

Usage:
    python audit_2d.py <keras_slices_data dir>

Prints, per split: number of slices, pixel share of each label, number of
slices containing each label, and the case IDs shared between splits
(expected: none). This audit found that labels 4 and 5 exist (an earlier
version of the code had mapped them to background) and that label 5, the
smallest class present in ~35% of slices, is the prostate (confirmed visually
with verify_2d_3d_alignment.py).
"""

import collections
import glob
import os
import sys

import nibabel as nib
import numpy as np

root = sys.argv[1]
for split in ["train", "validate", "test"]:
    files = sorted(glob.glob(os.path.join(root, f"keras_slices_seg_{split}", "*.nii.gz")))
    pix, slices = collections.Counter(), collections.Counter()
    for f in files:
        u, c = np.unique(nib.load(f).get_fdata().astype(int), return_counts=True)
        pix.update(dict(zip(u.tolist(), c.tolist())))
        slices.update(u.tolist())
    total = sum(pix.values())
    print(f"{split}: {len(files)} slices")
    print("  pixel share:", {k: f"{v / total:.4f}" for k, v in sorted(pix.items())})
    print("  slices containing label:", dict(sorted(slices.items())))

ids = {s: {os.path.basename(f).split("_")[1]
           for f in glob.glob(os.path.join(root, f"keras_slices_{s}", "*.nii.gz"))}
       for s in ["train", "validate", "test"]}
for a, b in [("train", "validate"), ("train", "test"), ("validate", "test")]:
    print(f"{a} / {b} shared cases: {sorted(ids[a] & ids[b])}")
