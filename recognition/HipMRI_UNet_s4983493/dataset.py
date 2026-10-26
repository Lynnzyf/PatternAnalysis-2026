"""
dataset.py — HipMRI 2D Slice Data Loader

Loads 2D MRI slices and corresponding segmentation masks from .nii.gz files.
Handles the filename prefix mismatch: images use 'case_' prefix, masks use 'seg_'.

Data structure on disk:
    keras_slices_train/       case_004_week_0_slice_0.nii.gz, ...
    keras_slices_seg_train/   seg_004_week_0_slice_0.nii.gz, ...
    keras_slices_validate/    (same pattern)
    keras_slices_seg_validate/
    keras_slices_test/
    keras_slices_seg_test/

Data audit (run over every mask file):
    - Slices: train 1545 / validate 660 / test 540
    - Segmentation labels: {0, 1, 2, 3, 4, 5} — 6 classes, all kept
    - Prostate label (default 5) covers ~0.3–0.5% of pixels and appears in
      only ~35% of slices -> severe class imbalance
    - Patient-level split verified: no case ID is shared between
      train / validate / test
"""

import os
import glob
import re
from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# nibabel is used only for loading .nii.gz files in the dataset loader,
# not in the model itself (modules.py is pure PyTorch)
import nibabel as nib

NUM_CLASSES = 6
TARGET_SHAPE = (256, 128)

_NAME_RE = re.compile(r"case_(\d+)_week_(\d+)_slice_(\d+)")


class HipMRIDataset(Dataset):
    """PyTorch Dataset for HipMRI 2D slices.

    Pairs each image slice with its segmentation mask by converting the
    filename prefix from 'case_' to 'seg_'.

    Args:
        image_dir: Path to directory containing image .nii.gz files
        mask_dir: Path to directory containing segmentation .nii.gz files
        transform: Optional callable for data augmentation on (image, mask) pairs
    """

    def __init__(self, image_dir: str, mask_dir: str,
                 transform: Optional[Callable] = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Collect all image file paths and sort for reproducibility
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))

        if len(self.image_files) == 0:
            raise FileNotFoundError(
                f"No .nii.gz files found in {image_dir}. "
                f"Check that the data path is correct."
            )

    def __len__(self) -> int:
        return len(self.image_files)

    def get_meta(self, idx: int) -> dict:
        """Parse (case, week, slice index) from the filename.

        Used later to rebuild each scan's slice order, so that base/apex
        (first/last prostate-containing) slices can be analysed separately.
        """
        m = _NAME_RE.search(os.path.basename(self.image_files[idx]))
        if m is None:
            return {"case": None, "week": None, "slice": None}
        return {"case": int(m.group(1)), "week": int(m.group(2)),
                "slice": int(m.group(3))}

    def _get_mask_path(self, image_path: str) -> str:
        """Convert image filename to corresponding mask filename.

        Image files:  case_004_week_0_slice_0.nii.gz
        Mask files:   seg_004_week_0_slice_0.nii.gz
        """
        basename = os.path.basename(image_path)
        suffix = basename.split("case_", 1)[1]
        return os.path.join(self.mask_dir, f"seg_{suffix}")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one image-mask pair.

        Returns:
            image: (1, H, W) float32 tensor, intensity-normalised
            mask:  (H, W) long tensor with integer class labels {0..5}
        """
        image_path = self.image_files[idx]
        mask_path = self._get_mask_path(image_path)

        image = nib.load(image_path).get_fdata().astype("float32")  # (H, W)
        mask = nib.load(mask_path).get_fdata().astype("float32")
        # NOTE: an earlier version mapped labels 4 and 5 to background here,
        # which silently removed the prostate from training. All 6 labels
        # are now kept.

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).long()

        # Resize to fixed size if needed (some slices are 256x144)
        if tuple(image_tensor.shape[-2:]) != TARGET_SHAPE:
            image_tensor = torch.nn.functional.interpolate(
                image_tensor.unsqueeze(0), size=TARGET_SHAPE,
                mode="bilinear", align_corners=False
            ).squeeze(0)
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor.float().unsqueeze(0).unsqueeze(0), size=TARGET_SHAPE,
                mode="nearest"
            ).long().squeeze(0).squeeze(0)

        # Intensity normalisation: zero-mean, unit-variance per slice
        mean = image_tensor.mean()
        std = image_tensor.std()
        if std > 0:
            image_tensor = (image_tensor - mean) / std

        return image_tensor, mask_tensor


class RandomAugmentation:
    """Anatomically plausible augmentation for pelvic MRI slices.

    Only a left-right mirror flip is used. Vertical flips and 180° rotations
    (used in an earlier version) are removed because they create impossible
    anatomy (e.g. rectum anterior to the bladder).

    Args:
        flip_prob: Probability of applying the left-right flip
        lr_axis: Array axis corresponding to the patient's left-right
            direction. In nibabel's (x, y) ordering x is usually left-right,
            i.e. axis 0 — verify visually before relying on this.
    """

    def __init__(self, flip_prob: float = 0.5, lr_axis: int = 0):
        self.flip_prob = flip_prob
        self.lr_axis = lr_axis

    def __call__(self, image, mask):
        if np.random.random() < self.flip_prob:
            image = np.flip(image, axis=self.lr_axis).copy()
            mask = np.flip(mask, axis=self.lr_axis).copy()
        return image, mask


def get_dataloaders(data_root: str, batch_size: int = 8,
                    num_workers: int = 4, augment: bool = True
                    ) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders.

    Uses the provided patient-level train/validate/test directories
    (verified to share no case IDs).
    """
    train_transform = RandomAugmentation() if augment else None

    train_dataset = HipMRIDataset(
        os.path.join(data_root, "keras_slices_train"),
        os.path.join(data_root, "keras_slices_seg_train"),
        transform=train_transform,
    )
    val_dataset = HipMRIDataset(
        os.path.join(data_root, "keras_slices_validate"),
        os.path.join(data_root, "keras_slices_seg_validate"),
    )
    test_dataset = HipMRIDataset(
        os.path.join(data_root, "keras_slices_test"),
        os.path.join(data_root, "keras_slices_seg_test"),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"Dataset sizes — Train: {len(train_dataset)}, "
          f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
