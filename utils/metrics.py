"""
Loss Functions & Evaluation Metrics for Segmentation
Includes: Dice Loss, BCE+Dice combined, IoU, Dice Coefficient
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1.  Dice Coefficient (metric, higher = better)
# ─────────────────────────────────────────────
def dice_coefficient(preds, targets, threshold=0.5, smooth=1e-6):
    """
    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        preds   : raw logits or probabilities  (B, 1, H, W)
        targets : binary ground-truth masks    (B, 1, H, W)
        threshold: binarisation threshold
        smooth  : small value to avoid division by zero

    Returns:
        Scalar Dice score (float between 0 and 1)
    """
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    intersection = (preds * targets).sum(dim=(2, 3))
    union        = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


# ─────────────────────────────────────────────
# 2.  IoU / Jaccard Index (metric, higher = better)
# ─────────────────────────────────────────────
def iou_score(preds, targets, threshold=0.5, smooth=1e-6):
    """
    IoU = |A ∩ B| / |A ∪ B|
    """
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    intersection = (preds * targets).sum(dim=(2, 3))
    union        = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


# ─────────────────────────────────────────────
# 3.  Dice Loss (loss, lower = better)
# ─────────────────────────────────────────────
class DiceLoss(nn.Module):
    """
    Dice Loss = 1 - Dice Coefficient
    Works great for class-imbalanced segmentation tasks
    (e.g. small lesions on large background).
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)

        intersection = (preds * targets).sum(dim=(2, 3))
        union        = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


# ─────────────────────────────────────────────
# 4.  BCE + Dice Combined Loss (recommended)
# ─────────────────────────────────────────────
class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice Loss.
    BCE ensures pixel-level accuracy; Dice handles class imbalance.

    Total Loss = α * BCE + (1 - α) * Dice
    Default α = 0.5 (equal weight)
    """
    def __init__(self, alpha=0.5, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.alpha     = alpha
        self.bce       = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, preds, targets):
        targets = targets.float()
        bce_val  = self.bce(preds, targets)
        dice_val = self.dice_loss(preds, targets)
        return self.alpha * bce_val + (1 - self.alpha) * dice_val


# ─────────────────────────────────────────────
# 5.  Pixel Accuracy (simple baseline metric)
# ─────────────────────────────────────────────
def pixel_accuracy(preds, targets, threshold=0.5):
    """
    Fraction of pixels correctly classified.
    """
    preds   = (torch.sigmoid(preds) > threshold).float()
    correct = (preds == targets.float()).sum()
    total   = targets.numel()
    return (correct / total).item()


# ─────────────────────────────────────────────
# 6.  Metric Tracker
# ─────────────────────────────────────────────
class MetricTracker:
    """Accumulates metrics over a full epoch and returns averages."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.loss  = 0.0
        self.dice  = 0.0
        self.iou   = 0.0
        self.acc   = 0.0
        self.count = 0

    def update(self, loss, preds, targets):
        self.loss  += loss
        self.dice  += dice_coefficient(preds, targets)
        self.iou   += iou_score(preds, targets)
        self.acc   += pixel_accuracy(preds, targets)
        self.count += 1

    def averages(self):
        n = max(self.count, 1)
        return {
            "loss" : round(self.loss / n, 4),
            "dice" : round(self.dice / n, 4),
            "iou"  : round(self.iou  / n, 4),
            "acc"  : round(self.acc  / n, 4),
        }


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    preds   = torch.randn(4, 1, 256, 256)
    targets = torch.randint(0, 2, (4, 1, 256, 256)).float()

    loss_fn = BCEDiceLoss()
    loss    = loss_fn(preds, targets)
    dice    = dice_coefficient(preds, targets)
    iou     = iou_score(preds, targets)

    print(f"BCEDice Loss : {loss.item():.4f}")
    print(f"Dice Coeff   : {dice:.4f}")
    print(f"IoU Score    : {iou:.4f}")
