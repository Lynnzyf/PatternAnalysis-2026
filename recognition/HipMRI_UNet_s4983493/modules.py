"""
modules.py — 2D U-Net for HipMRI Prostate Segmentation (baseline)

Implements a standard 2D U-Net (Ronneberger et al., 2015) with:
  - 4-level encoder-decoder architecture
  - Skip connections via concatenation
  - Dice + Cross-Entropy combined loss function
  - Evaluation helpers that accumulate Dice over a whole split
    (not averaged per batch)

No NumPy dependency — pure PyTorch as required by COMP3710 spec.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive 3x3 convolutions, each followed by BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """Encoder stage: DoubleConv followed by 2x2 max pooling.

    Returns both the feature map (for skip connection) and the pooled output.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        return features, self.pool(features)


class DecoderBlock(nn.Module):
    """Decoder stage: transposed-conv upsampling, skip concatenation, DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                      diff_h // 2, diff_h - diff_h // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Standard 2D U-Net.

    Args:
        in_channels: Number of input image channels (1 for greyscale MRI)
        num_classes: Number of segmentation classes (6 for HipMRI)
        base_filters: Number of filters in the first encoder level

    Input shape:  (B, in_channels, H, W)
    Output shape: (B, num_classes, H, W) — raw logits
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 6, base_filters: int = 64):
        super().__init__()
        f = base_filters

        self.enc1 = EncoderBlock(in_channels, f)
        self.enc2 = EncoderBlock(f, f * 2)
        self.enc3 = EncoderBlock(f * 2, f * 4)
        self.enc4 = EncoderBlock(f * 4, f * 8)

        self.bottleneck = DoubleConv(f * 8, f * 16)

        self.dec4 = DecoderBlock(f * 16, f * 8)
        self.dec3 = DecoderBlock(f * 8, f * 4)
        self.dec2 = DecoderBlock(f * 4, f * 2)
        self.dec1 = DecoderBlock(f * 2, f)

        self.final_conv = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        return self.final_conv(x)


class DiceLoss(nn.Module):
    """Soft Sørensen-Dice loss for multi-class segmentation.

    Args:
        smooth: Smoothing constant to prevent division by zero
        ignore_background: If True, exclude class 0 from the loss
    """

    def __init__(self, smooth: float = 1.0, ignore_background: bool = False):
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).float()

        start_class = 1 if self.ignore_background else 0
        dice_scores = []
        for c in range(start_class, num_classes):
            p = probs[:, c]
            t = targets_one_hot[:, c]
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            dice_scores.append((2.0 * intersection + self.smooth) / (union + self.smooth))

        return 1.0 - torch.stack(dice_scores).mean()


class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss (handles prostate class imbalance)."""

    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice_loss = DiceLoss(smooth=1.0, ignore_background=False)
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (self.dice_weight * self.dice_loss(logits, targets)
                + self.ce_weight * self.ce_loss(logits, targets.long()))


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------

def dice_counts(logits: torch.Tensor, targets: torch.Tensor,
                num_classes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-class overlap counts for one batch.

    Sum these over an entire split, then compute Dice once:
        dice_c = 2 * inter_c / (pred_c + target_c)
    This avoids two biases of per-batch averaging: batches with no prostate
    scoring a spurious 1.0, and unequal weighting of batches.

    Returns:
        inter, pred_sum, target_sum: each a (num_classes,) float tensor
    """
    preds = logits.argmax(dim=1)
    inter = torch.zeros(num_classes, device=logits.device)
    pred_sum = torch.zeros(num_classes, device=logits.device)
    target_sum = torch.zeros(num_classes, device=logits.device)
    for c in range(num_classes):
        p = preds == c
        t = targets == c
        inter[c] = (p & t).sum()
        pred_sum[c] = p.sum()
        target_sum[c] = t.sum()
    return inter, pred_sum, target_sum


def counts_to_dice(inter: torch.Tensor, pred_sum: torch.Tensor,
                   target_sum: torch.Tensor) -> torch.Tensor:
    """Convert accumulated counts to per-class Dice (0 if a class never occurs)."""
    return 2.0 * inter / (pred_sum + target_sum).clamp(min=1)


def per_slice_dice(logits: torch.Tensor, targets: torch.Tensor,
                   cls: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dice of one class for every slice in the batch.

    Returns:
        dice: (B,) per-slice Dice (meaningful only where gt_present is True)
        gt_present: (B,) bool, slice contains the class in ground truth
        pred_present: (B,) bool, model predicts the class somewhere in the slice
    """
    preds = logits.argmax(dim=1)
    p = (preds == cls).flatten(1).float()
    t = (targets == cls).flatten(1).float()
    inter = (p * t).sum(1)
    dice = 2.0 * inter / (p.sum(1) + t.sum(1)).clamp(min=1)
    return dice, t.sum(1) > 0, p.sum(1) > 0
