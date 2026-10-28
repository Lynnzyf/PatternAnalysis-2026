"""
dataset.py — 3D volume loading for HipMRI prostate segmentation

Data on disk:
    semantic_MRs/          B006_Week0_LFOV.nii.gz, ...      (X, Y, Z) = (256, 256, 128)
    semantic_labels_only/  B006_Week0_SEMANTIC.nii.gz, ...

Audit (all 211 volumes, 38 patients):
    - Labels {0..5} in every volume; prostate = label 5 (default, see
      --prostate_label), ~0.1% of voxels
    - 210 volumes are 256x256x128, one is 256x256x144
    - Voxel spacing 1.41-1.88 mm in-plane (X, Y), 1.56 mm along Z
    - Patient numbers 4-42 (no 34) are the same set as the 2D slice dataset

Axes after loading: (D, H, W) = (Z, X, Y). Z is the superior-inferior axis,
so the D slices are axial and the first/last prostate slices are the
base/apex. X (H) is the patient's left-right axis.

Split: patient-level and IDENTICAL to the 2D slice dataset
(val = patients 36-39, test = 40-42, rest = train), so a 2D model trained
on the 2D training split never sees a 3D test patient. All weekly scans of
a patient stay in the same split.

Resampling: in-plane 256x256 -> 128x128 (~3.4 mm), Z kept at 128 slices
(1.56 mm), because the base/apex analysis needs full superior-inferior
resolution. Preprocessed volumes are cached as .pt files.
"""

import glob
import os
import re

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

NUM_CLASSES = 6
DEFAULT_SHAPE = (128, 128, 128)          # (D, H, W) after resampling
VAL_PATIENTS = {36, 37, 38, 39}          # same as 2D keras_slices_validate
TEST_PATIENTS = {40, 41, 42}             # same as 2D keras_slices_test
_NAME_RE = re.compile(r"([A-Za-z]+)(\d+)_Week(\d+)")


def parse_name(path: str):
    """'B040_Week3_LFOV.nii.gz' -> {'patient': 'B040', 'number': 40, 'week': 3}."""
    m = _NAME_RE.match(os.path.basename(path))
    if m is None:
        return None
    return {"patient": m.group(1) + m.group(2), "number": int(m.group(2)),
            "week": int(m.group(3)), "name": os.path.basename(path).split("_LFOV")[0]}


def pair_volumes(image_dir: str, mask_dir: str) -> list:
    """Return [(image_path, mask_path, meta), ...] for every image with a mask."""
    pairs = []
    for img in sorted(glob.glob(os.path.join(image_dir, "*.nii.gz"))):
        meta = parse_name(img)
        if meta is None:
            print(f"[dataset] WARNING: cannot parse {os.path.basename(img)}, skipped")
            continue
        mask = os.path.join(mask_dir, os.path.basename(img).replace("_LFOV", "_SEMANTIC"))
        if not os.path.exists(mask):
            print(f"[dataset] WARNING: no mask for {os.path.basename(img)}, skipped")
            continue
        pairs.append((img, mask, meta))
    return pairs


def get_splits(image_dir: str, mask_dir: str) -> dict:
    """Patient-level split that mirrors the 2D dataset's split."""
    splits = {"train": [], "validate": [], "test": []}
    for p in pair_volumes(image_dir, mask_dir):
        n = p[2]["number"]
        key = "test" if n in TEST_PATIENTS else "validate" if n in VAL_PATIENTS else "train"
        splits[key].append(p)
    for key, items in splits.items():
        patients = sorted({m["patient"] for _, _, m in items})
        print(f"[dataset] {key}: {len(items)} volumes, {len(patients)} patients")
    return splits


def load_volume(path: str) -> np.ndarray:
    """Load a NIfTI volume and reorder (X, Y, Z) -> (Z, X, Y) = (D, H, W)."""
    return np.ascontiguousarray(
        np.transpose(np.asarray(nib.load(path).dataobj, dtype=np.float32), (2, 0, 1)))


def preprocess(image_path: str, mask_path: str, target_shape=DEFAULT_SHAPE):
    """Resample to target_shape and z-score the intensities.

    Returns image (D, H, W) float16 and mask (D, H, W) uint8.
    """
    image = torch.from_numpy(load_volume(image_path))[None, None]
    mask = torch.from_numpy(load_volume(mask_path))[None, None]
    image = F.interpolate(image, size=target_shape, mode="trilinear",
                          align_corners=False)[0, 0]
    mask = F.interpolate(mask, size=target_shape, mode="nearest")[0, 0]
    image = (image - image.mean()) / image.std().clamp(min=1e-6)
    return image.half(), mask.round().to(torch.uint8)


class HipMRI3DDataset(Dataset):
    """3D volumes with optional on-disk cache of the resampled tensors.

    Args:
        pairs: list from pair_volumes / get_splits
        target_shape: (D, H, W) after resampling
        cache_dir: folder for cached .pt volumes (None = no cache)
        augment: enable training augmentation
        lr_axis: axis of the (D, H, W) tensor that is the patient's
            left-right direction (1 = X). Only this axis is flipped.
    """

    def __init__(self, pairs, target_shape=DEFAULT_SHAPE, cache_dir=None,
                 augment=False, lr_axis=1):
        self.pairs = pairs
        self.target_shape = tuple(target_shape)
        self.cache_dir = cache_dir
        self.augment = augment
        self.lr_axis = lr_axis
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.pairs)

    def _cache_path(self, image_path):
        tag = "x".join(map(str, self.target_shape))
        name = os.path.basename(image_path).replace(".nii.gz", f"_{tag}.pt")
        return os.path.join(self.cache_dir, name)

    def _load(self, idx):
        image_path, mask_path, _ = self.pairs[idx]
        if self.cache_dir:
            cp = self._cache_path(image_path)
            if os.path.exists(cp):
                d = torch.load(cp)
                return d["image"], d["mask"]
        image, mask = preprocess(image_path, mask_path, self.target_shape)
        if self.cache_dir:
            torch.save({"image": image, "mask": mask}, cp)
        return image, mask

    def __getitem__(self, idx):
        image, mask = self._load(idx)
        image, mask = image.float(), mask.long()
        if self.augment:
            image, mask = self._augment(image, mask)
        return image.unsqueeze(0), mask

    def _augment(self, image, mask):
        """Anatomically plausible augmentation only.

        Left-right mirror (the pelvis is roughly symmetric), mild intensity
        scale/shift and noise. No superior-inferior or anterior-posterior
        flips: they would swap base/apex or put the rectum in front of the
        bladder.
        """
        if torch.rand(1).item() < 0.5:
            image, mask = image.flip(self.lr_axis), mask.flip(self.lr_axis)
        if torch.rand(1).item() < 0.5:
            image = image * torch.empty(1).uniform_(0.9, 1.1) \
                + torch.empty(1).uniform_(-0.1, 0.1)
        if torch.rand(1).item() < 0.3:
            image = image + 0.05 * torch.randn_like(image)
        return image, mask


def get_dataloaders(image_dir, mask_dir, batch_size=2, num_workers=2,
                    augment=True, target_shape=DEFAULT_SHAPE, cache_dir=None,
                    lr_axis=1):
    splits = get_splits(image_dir, mask_dir)
    make = lambda key, aug: HipMRI3DDataset(splits[key], target_shape, cache_dir,
                                            augment=aug, lr_axis=lr_axis)
    train_ds, val_ds, test_ds = make("train", augment), make("validate", False), make("test", False)
    kw = dict(num_workers=num_workers, pin_memory=True,
              persistent_workers=num_workers > 0)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **kw),
            DataLoader(val_ds, batch_size=1, shuffle=False, **kw),
            DataLoader(test_ds, batch_size=1, shuffle=False, **kw))
