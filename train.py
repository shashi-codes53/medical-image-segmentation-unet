"""
Training Script for U-Net Medical Image Segmentation
Features: Early stopping, LR scheduling, checkpoint saving, TensorBoard logging
"""

import os
import time
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.unet      import UNet
from data.dataset     import get_dataloaders
from utils.metrics    import BCEDiceLoss, MetricTracker


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
def get_config():
    parser = argparse.ArgumentParser(description="Train U-Net on Chest X-ray Segmentation")

    # Data
    parser.add_argument("--image_dir",  type=str, default="data/images", help="Path to images")
    parser.add_argument("--mask_dir",   type=str, default="data/masks",  help="Path to masks")
    parser.add_argument("--image_size", type=int, default=256)

    # Model
    parser.add_argument("--in_channels",  type=int, default=1, help="1 for grayscale, 3 for RGB")
    parser.add_argument("--out_channels", type=int, default=1, help="1 for binary segmentation")

    # Training
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--patience",   type=int,   default=10, help="Early stopping patience")

    # Output
    parser.add_argument("--save_dir",   type=str, default="outputs/checkpoints")
    parser.add_argument("--log_dir",    type=str, default="outputs/logs")

    return parser.parse_args()


# ─────────────────────────────────────────────
# One Epoch of Training
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    tracker = MetricTracker()

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks  = masks.float().to(device)

        # Forward pass
        preds = model(images)
        loss  = loss_fn(preds, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        optimizer.step()

        tracker.update(loss.item(), preds.detach(), masks.detach())

        if (batch_idx + 1) % 10 == 0:
            avg = tracker.averages()
            print(f"  Batch [{batch_idx+1}/{len(loader)}]  "
                  f"Loss: {avg['loss']:.4f}  Dice: {avg['dice']:.4f}")

    return tracker.averages()


# ─────────────────────────────────────────────
# One Epoch of Validation
# ─────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    tracker = MetricTracker()

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.float().to(device)

        preds = model(images)
        loss  = loss_fn(preds, masks)

        tracker.update(loss.item(), preds, masks)

    return tracker.averages()


# ─────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────
def train(cfg):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Train] Using device: {device}")

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        image_dir   = cfg.image_dir,
        mask_dir    = cfg.mask_dir,
        image_size  = cfg.image_size,
        batch_size  = cfg.batch_size,
        num_workers = 4 if device.type == "cuda" else 0
    )

    # Model
    model = UNet(in_channels=cfg.in_channels, out_channels=cfg.out_channels).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] U-Net  |  Parameters: {total_params:,}")

    # Loss, Optimizer, Scheduler
    loss_fn   = BCEDiceLoss(alpha=0.5)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )  # reduces LR when val Dice stops improving

    # Logging
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir,  exist_ok=True)
    writer = SummaryWriter(log_dir=cfg.log_dir)

    # Training state
    best_dice    = 0.0
    patience_cnt = 0
    history      = {"train": [], "val": []}

    print(f"\n{'='*60}")
    print(f"  Starting Training for {cfg.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # Train
        print(f"Epoch [{epoch}/{cfg.epochs}] — Training...")
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)

        # Validate
        print(f"Epoch [{epoch}/{cfg.epochs}] — Validating...")
        val_metrics = validate(model, val_loader, loss_fn, device)

        elapsed = time.time() - t0

        # Logging
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        writer.add_scalars("Loss", {"train": train_metrics["loss"], "val": val_metrics["loss"]}, epoch)
        writer.add_scalars("Dice", {"train": train_metrics["dice"], "val": val_metrics["dice"]}, epoch)
        writer.add_scalars("IoU",  {"train": train_metrics["iou"],  "val": val_metrics["iou"]},  epoch)

        print(f"\n{'─'*60}")
        print(f"  Epoch {epoch}/{cfg.epochs}  ({elapsed:.1f}s)")
        print(f"  Train → Loss: {train_metrics['loss']:.4f} | Dice: {train_metrics['dice']:.4f} | IoU: {train_metrics['iou']:.4f}")
        print(f"  Val   → Loss: {val_metrics['loss']:.4f}   | Dice: {val_metrics['dice']:.4f}   | IoU: {val_metrics['iou']:.4f}")
        print(f"{'─'*60}\n")

        # LR scheduler step
        scheduler.step(val_metrics["dice"])

        # Save best model
        if val_metrics["dice"] > best_dice:
            best_dice    = val_metrics["dice"]
            patience_cnt = 0
            ckpt_path    = os.path.join(cfg.save_dir, "best_model.pth")
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "best_dice"  : best_dice,
                "config"     : vars(cfg)
            }, ckpt_path)
            print(f"  ✅ New best Dice: {best_dice:.4f}  →  Saved to {ckpt_path}\n")
        else:
            patience_cnt += 1
            print(f"  ⏳ No improvement. Patience: {patience_cnt}/{cfg.patience}\n")

        # Early stopping
        if patience_cnt >= cfg.patience:
            print(f"  🛑 Early stopping triggered at epoch {epoch}.")
            break

    writer.close()
    print(f"\n✅ Training complete!  Best Dice: {best_dice:.4f}")
    print(f"   Best model saved at: {os.path.join(cfg.save_dir, 'best_model.pth')}")
    return history


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    cfg = get_config()
    train(cfg)
