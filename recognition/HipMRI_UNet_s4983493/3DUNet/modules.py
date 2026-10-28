"""
modules.py — 3D Improved U-Net (Isensee et al., 2017) and loss / metric helpers

Architecture, following the paper:
  - Context pathway: 3x3x3 conv (stride 2 between levels) followed by a
    context module = pre-activation residual block
    (IN-LReLU-conv3, dropout 0.3, IN-LReLU-conv3) added to its input
  - Localisation pathway: upsampling module (upsample x2 + conv3 halving
    the features), concatenation with the skip, localisation module
    (conv3 + conv1 halving the features)
  - Deep supervision: segmentation layers (1x1x1 convs) at three decoder
    levels, combined by upsampling and element-wise summation into the
    final output
  - Instance normalisation and leaky ReLU (slope 0.01) throughout;
    instance norm is also the right choice for batch size 2

Reference:
    Isensee, F., Kickingereder, P., Wick, W., Bendszus, M., Maier-Hein, K. H.
    (2017). Brain Tumor Segmentation and Radiomics Survival Prediction:
    Contribution to the BRATS 2017 Challenge. BrainLes @ MICCAI.

Pure PyTorch, no NumPy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(c: int) -> nn.Module:
    return nn.InstanceNorm3d(c, affine=True)


class ConvNormAct(nn.Sequential):
    """Conv3d -> InstanceNorm -> LeakyReLU."""

    def __init__(self, cin: int, cout: int, k: int = 3):
        super().__init__(
            nn.Conv3d(cin, cout, k, padding=k // 2, bias=False),
            _norm(cout),
            nn.LeakyReLU(0.01, inplace=True),
        )


class ContextModule(nn.Module):
    """Pre-activation residual block with dropout between the two convs."""

    def __init__(self, c: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            _norm(c), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(c, c, 3, padding=1, bias=False),
            nn.Dropout3d(dropout),
            _norm(c), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(c, c, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class UpsamplingModule(nn.Module):
    """Upsample to the skip's size, then conv3 to the lower channel count."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = ConvNormAct(cin, cout, 3)

    def forward(self, x: torch.Tensor, size) -> torch.Tensor:
        return self.conv(F.interpolate(x, size=size, mode="nearest"))


class LocalisationModule(nn.Sequential):
    """conv3 on the concatenated features, then conv1 halving them (2c -> c)."""

    def __init__(self, c: int):
        super().__init__(ConvNormAct(2 * c, 2 * c, 3), ConvNormAct(2 * c, c, 1))


class ImprovedUNet3D(nn.Module):
    """3D Improved U-Net.

    Args:
        in_channels: input channels (1 for MRI)
        num_classes: output classes (6 for HipMRI)
        base_filters: filters at the first level (16 in the paper)
        dropout: dropout inside the context modules (0.3 in the paper)

    Input (B, in_channels, D, H, W) -> logits (B, num_classes, D, H, W).
    D, H, W should be divisible by 16.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 6,
                 base_filters: int = 16, dropout: float = 0.3):
        super().__init__()
        f = base_filters
        ch = [f, 2 * f, 4 * f, 8 * f, 16 * f]

        self.in_conv = nn.Conv3d(in_channels, f, 3, padding=1, bias=False)
        self.down = nn.ModuleList(
            nn.Conv3d(ch[i], ch[i + 1], 3, stride=2, padding=1, bias=False)
            for i in range(4))
        self.context = nn.ModuleList(ContextModule(c, dropout) for c in ch)

        # decoder runs over levels 3, 2, 1, 0
        self.up = nn.ModuleList(UpsamplingModule(ch[l + 1], ch[l]) for l in (3, 2, 1, 0))
        self.loc = nn.ModuleList(LocalisationModule(ch[l]) for l in (3, 2, 1))
        self.final = ConvNormAct(2 * f, 2 * f, 3)

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


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Soft multi-class Dice loss, averaged over all classes."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        probs = F.softmax(logits.float(), dim=1)
        one_hot = F.one_hot(targets, num_classes).movedim(-1, 1).float()
        dims = tuple(d for d in range(probs.dim()) if d != 1)
        inter = (probs * one_hot).sum(dims)
        union = probs.sum(dims) + one_hot.sum(dims)
        return 1.0 - ((2 * inter + self.smooth) / (union + self.smooth)).mean()


class DiceCELoss(nn.Module):
    """0.5 * Dice + 0.5 * cross-entropy (same loss as the 2D baseline).

    No label clamping: a label >= num_classes raises an error instead of
    being silently merged into another class.
    """

    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight, self.ce_weight = dice_weight, ce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (self.dice_weight * self.dice(logits, targets)
                + self.ce_weight * self.ce(logits.float(), targets))


# --------------------------------------------------------------------------
# Metrics (accumulate over a whole split, then compute once)
# --------------------------------------------------------------------------

def dice_counts(logits: torch.Tensor, targets: torch.Tensor, num_classes: int):
    """Per-class (intersection, predicted, target) voxel counts for one batch."""
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
    return 2.0 * inter / (pred_sum + target_sum).clamp(min=1)


def per_sample_dice(logits: torch.Tensor, targets: torch.Tensor, cls: int) -> torch.Tensor:
    """Dice of one class for each sample (volume) in the batch."""
    preds = logits.argmax(dim=1)
    p = (preds == cls).flatten(1).float()
    t = (targets == cls).flatten(1).float()
    return 2.0 * (p * t).sum(1) / (p.sum(1) + t.sum(1)).clamp(min=1)
