"""
modules.py — model components, loss and metrics for HipMRI prostate segmentation

Models
  - UNet2D          : baseline 2D U-Net (Ronneberger et al., 2015), applied to
                      coronal slices
  - ImprovedUNet3D  : 3D Improved U-Net (Isensee et al., 2017) on full volumes

Shared
  - DiceLoss / DiceCELoss : soft multi-class Dice + cross-entropy, works for
                            2D (B, C, H, W) and 3D (B, C, D, H, W) logits
  - dice_counts / counts_to_dice / per_sample_dice : evaluation helpers that
    accumulate overlap counts over a whole split before computing Dice

Pure PyTorch, no NumPy (as required by the task specification).

References
  Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks
  for Biomedical Image Segmentation. MICCAI.
  Isensee, F., Kickingereder, P., Wick, W., Bendszus, M., Maier-Hein, K. H.
  (2017). Brain Tumor Segmentation and Radiomics Survival Prediction:
  Contribution to the BRATS 2017 Challenge. BrainLes @ MICCAI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 2D U-Net (baseline)
# ===========================================================================

class DoubleConv2D(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) x 2."""

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


class EncoderBlock2D(nn.Module):
    """DoubleConv followed by 2x2 max pooling; returns (skip features, pooled)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv2D(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        features = self.conv(x)
        return features, self.pool(features)


class DecoderBlock2D(nn.Module):
    """Transposed-conv upsampling, skip concatenation, DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv2D(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                      diff_h // 2, diff_h - diff_h // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet2D(nn.Module):
    """Standard 4-level 2D U-Net.

    Input (B, in_channels, H, W) -> logits (B, num_classes, H, W).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 6, base_filters: int = 64):
        super().__init__()
        f = base_filters
        self.enc1 = EncoderBlock2D(in_channels, f)
        self.enc2 = EncoderBlock2D(f, f * 2)
        self.enc3 = EncoderBlock2D(f * 2, f * 4)
        self.enc4 = EncoderBlock2D(f * 4, f * 8)
        self.bottleneck = DoubleConv2D(f * 8, f * 16)
        self.dec4 = DecoderBlock2D(f * 16, f * 8)
        self.dec3 = DecoderBlock2D(f * 8, f * 4)
        self.dec2 = DecoderBlock2D(f * 4, f * 2)
        self.dec1 = DecoderBlock2D(f * 2, f)
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


# ===========================================================================
# 3D Improved U-Net (Isensee et al., 2017)
# ===========================================================================

def _norm3d(c: int) -> nn.Module:
    return nn.InstanceNorm3d(c, affine=True)


class ConvNormAct3D(nn.Sequential):
    """Conv3d -> InstanceNorm -> LeakyReLU(0.01)."""

    def __init__(self, cin: int, cout: int, k: int = 3):
        super().__init__(
            nn.Conv3d(cin, cout, k, padding=k // 2, bias=False),
            _norm3d(cout),
            nn.LeakyReLU(0.01, inplace=True),
        )


class ContextModule3D(nn.Module):
    """Pre-activation residual block: (IN-LReLU-conv3), dropout, (IN-LReLU-conv3), + input."""

    def __init__(self, c: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            _norm3d(c), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(c, c, 3, padding=1, bias=False),
            nn.Dropout3d(dropout),
            _norm3d(c), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(c, c, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class UpsamplingModule3D(nn.Module):
    """Upsample to the skip's size, then conv3 to the lower channel count."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = ConvNormAct3D(cin, cout, 3)

    def forward(self, x: torch.Tensor, size) -> torch.Tensor:
        return self.conv(F.interpolate(x, size=size, mode="nearest"))


class LocalisationModule3D(nn.Sequential):
    """conv3 on the concatenated features, then conv1 halving them (2c -> c)."""

    def __init__(self, c: int):
        super().__init__(ConvNormAct3D(2 * c, 2 * c, 3), ConvNormAct3D(2 * c, c, 1))


class ImprovedUNet3D(nn.Module):
    """3D Improved U-Net with context/localisation modules and deep supervision.

    Segmentation layers at three decoder levels are upsampled and summed
    element-wise into the final output (deep supervision as in the paper).

    Input (B, in_channels, D, H, W) -> logits (B, num_classes, D, H, W);
    D, H, W should be divisible by 16.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 6,
                 base_filters: int = 16, dropout: float = 0.3):
        super().__init__()
        f = base_filters
        ch = [f, 2 * f, 4 * f, 8 * f, 16 * f]

        self.in_conv = nn.Conv3d(in_channels, f, 3, padding=1, bias=False)
        self.down = nn.ModuleList(
            nn.Conv3d(ch[i], ch[i + 1], 3, stride=2, padding=1, bias=False) for i in range(4))
        self.context = nn.ModuleList(ContextModule3D(c, dropout) for c in ch)

        # decoder runs over levels 3, 2, 1, 0
        self.up = nn.ModuleList(UpsamplingModule3D(ch[l + 1], ch[l]) for l in (3, 2, 1, 0))
        self.loc = nn.ModuleList(LocalisationModule3D(ch[l]) for l in (3, 2, 1))
        self.final = ConvNormAct3D(2 * f, 2 * f, 3)

        # segmentation layers at levels 2, 1, 0
        self.seg = nn.ModuleList([
            nn.Conv3d(ch[2], num_classes, 1),
            nn.Conv3d(ch[1], num_classes, 1),
            nn.Conv3d(2 * f, num_classes, 1),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context[0](self.in_conv(x))
        skips = [x]
        for i in range(4):
            x = self.context[i + 1](self.down[i](x))
            skips.append(x)

        seg_sum = None
        for j, level in enumerate((3, 2, 1, 0)):
            skip = skips[level]
            x = torch.cat([skip, self.up[j](x, skip.shape[2:])], dim=1)
            x = self.loc[j](x) if level > 0 else self.final(x)
            if level <= 2:
                s = self.seg[2 - level](x)
                if seg_sum is not None:
                    s = s + F.interpolate(seg_sum, size=s.shape[2:],
                                          mode="trilinear", align_corners=False)
                seg_sum = s
        return seg_sum


# ===========================================================================
# Loss (shared by both models)
# ===========================================================================

class DiceLoss(nn.Module):
    """Soft multi-class Dice loss over the whole batch, averaged over classes."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        probs = F.softmax(logits.float(), dim=1)
        one_hot = F.one_hot(targets.long(), num_classes).movedim(-1, 1).float()
        dims = tuple(d for d in range(probs.dim()) if d != 1)
        inter = (probs * one_hot).sum(dims)
        union = probs.sum(dims) + one_hot.sum(dims)
        return 1.0 - ((2 * inter + self.smooth) / (union + self.smooth)).mean()


class DiceCELoss(nn.Module):
    """0.5 * Dice + 0.5 * cross-entropy.

    No label clamping: a label >= num_classes raises an error instead of being
    silently merged into another class.
    """

    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight, self.ce_weight = dice_weight, ce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (self.dice_weight * self.dice(logits, targets)
                + self.ce_weight * self.ce(logits.float(), targets.long()))


# ===========================================================================
# Metrics (accumulate over a whole split, then compute once)
# ===========================================================================

def dice_counts(logits: torch.Tensor, targets: torch.Tensor, num_classes: int):
    """Per-class (intersection, predicted, target) counts for one batch."""
    preds = logits.argmax(dim=1)
    inter = torch.zeros(num_classes, device=logits.device)
    pred_sum = torch.zeros(num_classes, device=logits.device)
    target_sum = torch.zeros(num_classes, device=logits.device)
    for c in range(num_classes):
        p, t = preds == c, targets == c
        inter[c] = (p & t).sum()
        pred_sum[c] = p.sum()
        target_sum[c] = t.sum()
    return inter, pred_sum, target_sum


def counts_to_dice(inter, pred_sum, target_sum) -> torch.Tensor:
    """Per-class Dice from accumulated counts (0 if a class never occurs)."""
    return 2.0 * inter / (pred_sum + target_sum).clamp(min=1)


def per_sample_dice(logits: torch.Tensor, targets: torch.Tensor, cls: int):
    """Dice of one class for every sample (slice in 2D, volume in 3D).

    Returns:
        dice: (B,) Dice per sample (meaningful where gt_present)
        gt_present: (B,) sample contains the class in the ground truth
        pred_present: (B,) the model predicts the class somewhere in the sample
    """
    preds = logits.argmax(dim=1)
    p = (preds == cls).flatten(1).float()
    t = (targets == cls).flatten(1).float()
    dice = 2.0 * (p * t).sum(1) / (p.sum(1) + t.sum(1)).clamp(min=1)
    return dice, t.sum(1) > 0, p.sum(1) > 0
