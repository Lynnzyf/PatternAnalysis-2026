"""
dataset.py — data loading for HipMRI prostate segmentation (2D slices and 3D volumes)

Labels (audited over every mask, both datasets): 0 background, 1 body, 2 bone,
3 bladder, 4 rectum, 5 prostate. The prostate is ~0.1-0.5% of voxels.

2D data (keras_slices_data)
  keras_slices_{train,validate,test}/      case_004_week_0_slice_0.nii.gz ...
  keras_slices_seg_{train,validate,test}/  seg_004_week_0_slice_0.nii.gz ...
  - 11460 / 660 / 540 slices; every volume contributes exactly 60 slices
  - The slices are CORONAL slices (left-right x superior-inferior) of the 3D
    volumes below, covering a ~10 cm anterior-posterior slab around the
    prostate (verified by image correlation with patient 040)
  - Shape 256x128 (one volume 256x144, resized to 256x128)

3D data (semantic_MRs / semantic_labels_only)
  B040_Week0_LFOV.nii.gz / B040_Week0_SEMANTIC.nii.gz ...
  - 211 volumes, 38 patients, 256x256x128 (one 256x256x144)
  - Loaded as (D, H, W) = (Z, X, Y): D superior-inferior (axial slices),
    H left-right, W anterior-posterior

Split (both datasets, patient level): the 2D dataset's provided split is
used, and the 3D split mirrors it (validation = patients 36-39, test = 40-42),
verified to share no patient between splits. So the 2D model never sees a 3D
test patient, and every weekly scan of a patient stays in one split.

NumPy is used here only for loading and augmentation (allowed outside
modules.py).
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
PROSTATE_LABEL = 5
BLADDER_LABEL = 3


# ===========================================================================
# 2D coronal slices
# ===========================================================================

SHAPE_2D = (256, 128)
_SLICE_RE = re.compile(r"case_(\d+)_week_(\d+)_slice_(\d+)")


class HipMRI2DDataset(Dataset):
    """2D slices paired with masks by name (case_X... -> seg_X...).

    Each item: image (1, 256, 128) float32, z-scored per slice;
    mask (256, 128) long.
    """

    def __init__(self, image_dir: str, mask_dir: str, transform=None):
        self.mask_dir = mask_dir
        self.transform = transform
        images = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
        if not images:
            raise FileNotFoundError(f"No .nii.gz files in {image_dir}")
        masks = {os.path.basename(p) for p in glob.glob(os.path.join(mask_dir, "*.nii.gz"))}
        self.image_files = [p for p in images
                            if os.path.basename(self._mask_path(p)) in masks]
        skipped = len(images) - len(self.image_files)
        if skipped:
            print(f"[dataset] WARNING: {skipped} images in {image_dir} have no mask "
                  f"and are skipped")
        if len(masks) > len(self.image_files):
            print(f"[dataset] WARNING: {len(masks) - len(self.image_files)} masks in "
                  f"{mask_dir} have no image")

    def __len__(self):
        return len(self.image_files)

    def _mask_path(self, image_path):
        suffix = os.path.basename(image_path).split("case_", 1)[1]
        return os.path.join(self.mask_dir, f"seg_{suffix}")

    def get_meta(self, idx: int) -> dict:
        """(case, week, slice) parsed from the filename."""
        m = _SLICE_RE.search(os.path.basename(self.image_files[idx]))
        if m is None:
            return {"case": None, "week": None, "slice": None}
        return {"case": int(m.group(1)), "week": int(m.group(2)), "slice": int(m.group(3))}

    def __getitem__(self, idx):
        path = self.image_files[idx]
        image = nib.load(path).get_fdata().astype(np.float32)
        mask = nib.load(self._mask_path(path)).get_fdata().astype(np.float32)
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        image = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
        mask = torch.from_numpy(np.ascontiguousarray(mask)).long()
        if tuple(image.shape[-2:]) != SHAPE_2D:
            image = F.interpolate(image[None], size=SHAPE_2D, mode="bilinear",
                                  align_corners=False)[0]
            mask = F.interpolate(mask[None, None].float(), size=SHAPE_2D,
                                 mode="nearest").long()[0, 0]
        std = image.std()
        if std > 0:
            image = (image - image.mean()) / std
        return image, mask


class RandomFlip2D:
    """Left-right mirror flip only (axis 0 of a coronal slice is left-right).

    Vertical flips / 180-degree rotations are not used: they produce
    anatomically impossible images.
    """

    def __init__(self, p: float = 0.5, lr_axis: int = 0):
        self.p, self.lr_axis = p, lr_axis

    def __call__(self, image, mask):
        if np.random.random() < self.p:
            image = np.flip(image, axis=self.lr_axis).copy()
            mask = np.flip(mask, axis=self.lr_axis).copy()
        return image, mask


def get_dataloaders_2d(data_root, batch_size=16, num_workers=2, augment=True):
    ds = lambda split, tf: HipMRI2DDataset(
        os.path.join(data_root, f"keras_slices_{split}"),
        os.path.join(data_root, f"keras_slices_seg_{split}"), transform=tf)
    train_ds = ds("train", RandomFlip2D() if augment else None)
    val_ds, test_ds = ds("validate", None), ds("test", None)
    print(f"[dataset] 2D slices — train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)}")
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (DataLoader(train_ds, shuffle=True, drop_last=True, **kw),
            DataLoader(val_ds, shuffle=False, **kw),
            DataLoader(test_ds, shuffle=False, **kw))


# ===========================================================================
# 3D volumes
# ===========================================================================

SHAPE_3D = (128, 256, 256)               # native grid (D, H, W)
VAL_PATIENTS = {36, 37, 38, 39}          # same patients as keras_slices_validate
TEST_PATIENTS = {40, 41, 42}             # same patients as keras_slices_test
_VOLUME_RE = re.compile(r"([A-Za-z]+)(\d+)_Week(\d+)")


def parse_name_3d(path: str):
    """'B040_Week3_LFOV.nii.gz' -> {'patient': 'B040', 'number': 40, 'week': 3, 'name': 'B040_Week3'}."""
    m = _VOLUME_RE.match(os.path.basename(path))
    if m is None:
        return None
    return {"patient": m.group(1) + m.group(2), "number": int(m.group(2)),
            "week": int(m.group(3)), "name": os.path.basename(path).split("_LFOV")[0]}


def get_splits_3d(image_dir: str, mask_dir: str) -> dict:
    """Patient-level split mirroring the 2D dataset: {split: [(image, mask, meta), ...]}."""
    splits = {"train": [], "validate": [], "test": []}
    for img in sorted(glob.glob(os.path.join(image_dir, "*.nii.gz"))):
        meta = parse_name_3d(img)
        if meta is None:
            print(f"[dataset] WARNING: cannot parse {os.path.basename(img)}, skipped")
            continue
        mask = os.path.join(mask_dir, os.path.basename(img).replace("_LFOV", "_SEMANTIC"))
        if not os.path.exists(mask):
            print(f"[dataset] WARNING: no mask for {os.path.basename(img)}, skipped")
            continue
        n = meta["number"]
        key = "test" if n in TEST_PATIENTS else "validate" if n in VAL_PATIENTS else "train"
        splits[key].append((img, mask, meta))
    for key, items in splits.items():
        patients = {m["patient"] for _, _, m in items}
        print(f"[dataset] 3D {key}: {len(items)} volumes, {len(patients)} patients")
    return splits


def load_volume_3d(path: str) -> np.ndarray:
    """Load a NIfTI volume and reorder (X, Y, Z) -> (Z, X, Y) = (D, H, W)."""
    return np.ascontiguousarray(
        np.transpose(np.asarray(nib.load(path).dataobj, dtype=np.float32), (2, 0, 1)))


def preprocess_3d(image_path: str, mask_path: str, target_shape=SHAPE_3D):
    """Resample to target_shape and z-score; returns image float16, mask uint8."""
    image = torch.from_numpy(load_volume_3d(image_path))[None, None]
    mask = torch.from_numpy(load_volume_3d(mask_path))[None, None]
    if tuple(image.shape[2:]) != tuple(target_shape):
        image = F.interpolate(image, size=target_shape, mode="trilinear", align_corners=False)
        mask = F.interpolate(mask, size=target_shape, mode="nearest")
    image, mask = image[0, 0], mask[0, 0]
    image = (image - image.mean()) / image.std().clamp(min=1e-6)
    return image.half(), mask.round().to(torch.uint8)


class HipMRI3DDataset(Dataset):
    """3D volumes with an optional on-disk cache of the preprocessed tensors.

    Each item: image (1, D, H, W) float32, mask (D, H, W) long.
    Augmentation: left-right flip (axis 1 = H) and mild intensity scale /
    shift / noise. No superior-inferior or anterior-posterior flips (they would
    swap base and apex or put the rectum in front of the bladder).
    """

    def __init__(self, pairs, target_shape=SHAPE_3D, cache_dir=None, augment=False,
                 lr_axis=1):
        self.pairs = pairs
        self.target_shape = tuple(target_shape)
        self.cache_dir = cache_dir
        self.augment = augment
        self.lr_axis = lr_axis
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.pairs)

    def _load(self, idx):
        image_path, mask_path, _ = self.pairs[idx]
        cache = None
        if self.cache_dir:
            tag = "x".join(map(str, self.target_shape))
            cache = os.path.join(self.cache_dir, os.path.basename(image_path)
                                 .replace(".nii.gz", f"_{tag}.pt"))
            if os.path.exists(cache):
                d = torch.load(cache)
                return d["image"], d["mask"]
        image, mask = preprocess_3d(image_path, mask_path, self.target_shape)
        if cache:
            torch.save({"image": image, "mask": mask}, cache)
        return image, mask

    def __getitem__(self, idx):
        image, mask = self._load(idx)
        image, mask = image.float(), mask.long()
        if self.augment:
            if torch.rand(1).item() < 0.5:
                image, mask = image.flip(self.lr_axis), mask.flip(self.lr_axis)
            if torch.rand(1).item() < 0.5:
                image = image * torch.empty(1).uniform_(0.9, 1.1) \
                    + torch.empty(1).uniform_(-0.1, 0.1)
            if torch.rand(1).item() < 0.3:
                image = image + 0.05 * torch.randn_like(image)
        return image.unsqueeze(0), mask


def get_dataloaders_3d(image_dir, mask_dir, batch_size=2, num_workers=2, augment=True,
                       target_shape=SHAPE_3D, cache_dir=None):
    splits = get_splits_3d(image_dir, mask_dir)
    ds = lambda key, aug: HipMRI3DDataset(splits[key], target_shape, cache_dir, augment=aug)
    kw = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    return (DataLoader(ds("train", augment), batch_size=batch_size, shuffle=True,
                       drop_last=True, **kw),
            DataLoader(ds("validate", False), batch_size=1, shuffle=False, **kw),
            DataLoader(ds("test", False), batch_size=1, shuffle=False, **kw))
