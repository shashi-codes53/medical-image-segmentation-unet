"""
Inference & Visualization Script
- Load trained model
- Run predictions on test images
- Visualize: original | ground truth | prediction | overlay
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from PIL import Image

import torch
import torchvision.transforms.functional as TF

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.unet   import UNet
from utils.metrics import dice_coefficient, iou_score


# ─────────────────────────────────────────────
# Load trained model from checkpoint
# ─────────────────────────────────────────────
def load_model(checkpoint_path, in_channels=1, out_channels=1, device="cpu"):
    model = UNet(in_channels=in_channels, out_channels=out_channels)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"[Inference] Loaded model from: {checkpoint_path}")
    print(f"            Checkpoint epoch : {ckpt.get('epoch', '?')}")
    print(f"            Best Dice        : {ckpt.get('best_dice', '?'):.4f}")
    return model


# ─────────────────────────────────────────────
# Preprocess a single image for inference
# ─────────────────────────────────────────────
def preprocess(image_path, image_size=256):
    image = Image.open(image_path).convert("L")
    image = TF.resize(image, [image_size, image_size])
    image = TF.to_tensor(image)
    image = TF.normalize(image, mean=[0.5], std=[0.5])
    return image.unsqueeze(0)  # add batch dimension: (1, 1, H, W)


# ─────────────────────────────────────────────
# Run inference on a single image
# ─────────────────────────────────────────────
@torch.no_grad()
def predict(model, image_tensor, device, threshold=0.5):
    image_tensor = image_tensor.to(device)
    logits = model(image_tensor)
    probs  = torch.sigmoid(logits)
    mask   = (probs > threshold).float()
    return probs.squeeze().cpu().numpy(), mask.squeeze().cpu().numpy()


# ─────────────────────────────────────────────
# Visualize results for one sample
# ─────────────────────────────────────────────
def visualize_prediction(image_path, mask_path, model, device, image_size=256,
                          save_path=None, threshold=0.5):
    """
    Plots a 2x2 grid:
      [Original X-ray]  [Ground Truth Mask]
      [Predicted Mask]  [Overlay on X-ray]
    """
    # Preprocess
    img_tensor = preprocess(image_path, image_size)
    probs, pred_mask = predict(model, img_tensor, device, threshold)

    # Load originals for display
    orig_img  = np.array(Image.open(image_path).convert("L").resize((image_size, image_size)))
    gt_mask   = np.array(Image.open(mask_path).convert("L").resize((image_size, image_size),
                          Image.NEAREST))
    gt_binary = (gt_mask > 127).astype(np.uint8)

    # Compute metrics vs ground truth
    pred_t = torch.tensor(pred_mask).unsqueeze(0).unsqueeze(0)
    gt_t   = torch.tensor(gt_binary.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    logits = torch.logit(torch.clamp(pred_t, 1e-6, 1 - 1e-6))
    dice   = dice_coefficient(logits, gt_t, threshold)
    iou    = iou_score(logits, gt_t, threshold)

    # Overlay: green = correct, red = false positive, blue = false negative
    overlay = np.stack([orig_img, orig_img, orig_img], axis=-1).copy()
    pred_b  = pred_mask > threshold
    tp = pred_b & gt_binary.astype(bool)
    fp = pred_b & ~gt_binary.astype(bool)
    fn = ~pred_b & gt_binary.astype(bool)
    overlay[tp, 1] = 200  # green
    overlay[fp, 0] = 200  # red
    overlay[fn, 2] = 200  # blue

    # Plot
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(f"U-Net Segmentation  |  Dice: {dice:.3f}  |  IoU: {iou:.3f}",
                 fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.05)

    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    titles  = ["Original X-ray", "Ground Truth", "Prediction", "Overlay"]
    images  = [orig_img, gt_binary * 255, pred_mask * 255, overlay]
    cmaps   = ["gray", "gray", "inferno", None]

    for ax, title, img, cmap in zip(axes, titles, images, cmaps):
        if cmap:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=255 if img.max() > 1 else 1)
        else:
            ax.imshow(img.astype(np.uint8))
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    # Legend for overlay
    from matplotlib.patches import Patch
    legend = [Patch(color=(0, 0.78, 0), label="True Positive"),
              Patch(color=(0.78, 0, 0), label="False Positive"),
              Patch(color=(0, 0, 0.78), label="False Negative")]
    axes[-1].legend(handles=legend, loc="lower right", fontsize=8,
                    framealpha=0.7, edgecolor="white")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] Visualization → {save_path}")
    else:
        plt.show()

    plt.close()
    return {"dice": dice, "iou": iou}


# ─────────────────────────────────────────────
# Batch evaluation on test set
# ─────────────────────────────────────────────
def evaluate_test_set(model, test_loader, device, threshold=0.5):
    from utils.metrics import MetricTracker, BCEDiceLoss
    model.eval()
    tracker = MetricTracker()
    loss_fn = BCEDiceLoss()

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks  = masks.float().to(device)
            preds  = model(images)
            loss   = loss_fn(preds, masks)
            tracker.update(loss.item(), preds, masks)

    results = tracker.averages()
    print("\n" + "="*40)
    print("  Test Set Results")
    print("="*40)
    for k, v in results.items():
        print(f"  {k.upper():6s}: {v:.4f}")
    print("="*40)
    return results


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth")
    parser.add_argument("--image",      type=str, required=True, help="Path to a test image")
    parser.add_argument("--mask",       type=str, required=True, help="Path to corresponding mask")
    parser.add_argument("--save",       type=str, default=None,  help="Path to save visualization")
    parser.add_argument("--size",       type=int, default=256)
    parser.add_argument("--threshold",  type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.checkpoint, device=device)

    metrics = visualize_prediction(
        image_path=args.image,
        mask_path=args.mask,
        model=model,
        device=device,
        image_size=args.size,
        save_path=args.save,
        threshold=args.threshold,
    )
    print(f"\nDice: {metrics['dice']:.4f}  |  IoU: {metrics['iou']:.4f}")
