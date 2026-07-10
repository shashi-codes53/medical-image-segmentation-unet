"""
Dataset & DataLoader for NIH ChestX-ray14
Handles loading, preprocessing, and augmentation.
"""

import os
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


# ─────────────────────────────────────────────
# 1.  Custom Augmentation (image + mask together)
# ─────────────────────────────────────────────
class JointTransform:
    """
    Applies the SAME random transforms to both image and mask.
    This is critical — mask must mirror every spatial change to the image.
    """
    def __init__(self, image_size=256, augment=True):
        self.image_size = image_size
        self.augment = augment

    def __call__(self, image, mask):
        # Resize both
        image = TF.resize(image, [self.image_size, self.image_size])
        mask  = TF.resize(mask,  [self.image_size, self.image_size],
                          interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask  = TF.vflip(mask)

            # Random rotation (-15 to +15 degrees)
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask  = TF.rotate(mask,  angle)

            # Intensity jitter — ONLY on image, never on mask
            image = T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1
            )(image)

            # Elastic deformation (using OpenCV)
            image_np = np.array(image)
            mask_np  = np.array(mask)
            image_np, mask_np = elastic_transform(image_np, mask_np, alpha=34, sigma=4)
            image = Image.fromarray(image_np)
            mask  = Image.fromarray(mask_np)

        # Convert to tensor
        image = TF.to_tensor(image)            # shape: (C, H, W), values [0,1]
        mask  = torch.from_numpy(
            np.array(mask)
        ).long().unsqueeze(0)                  # shape: (1, H, W)

        # Normalize image (ImageNet stats work for grayscale too)
        image = TF.normalize(image, mean=[0.5], std=[0.5])

        return image, mask


def elastic_transform(image, mask, alpha=34, sigma=4, random_state=None):
    """
    Elastic deformation of images as described in Simard et al. 2003.
    Helps the U-Net generalise to shape variations in medical images.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape[:2]
    dx = cv2.GaussianBlur(
        (random_state.rand(*shape) * 2 - 1).astype(np.float32), (0, 0), sigma
    ) * alpha
    dy = cv2.GaussianBlur(
        (random_state.rand(*shape) * 2 - 1).astype(np.float32), (0, 0), sigma
    ) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    image_out = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)
    mask_out  = cv2.remap(mask,  map_x, map_y, interpolation=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_REFLECT_101)
    return image_out, mask_out


# ─────────────────────────────────────────────
# 2.  Dataset Class
# ─────────────────────────────────────────────
class ChestXrayDataset(Dataset):
    """
    NIH ChestX-ray14 Dataset loader.

    Expected folder structure:
        data/
          images/       ← .png chest X-ray images
          masks/        ← binary .png masks (same filename as image)

    Download instructions:
        https://nihcc.app.box.com/v/ChestXray-NIHCC
        Or use: https://www.kaggle.com/datasets/nih-chest-xrays/data

    Args:
        image_dir  : path to images folder
        mask_dir   : path to masks folder
        transform  : JointTransform instance
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.transform = transform

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        print(f"[Dataset] Found {len(self.images)} images in {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name  = self.images[idx]
        img_path  = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir,  img_name)

        # Load as grayscale (L mode) for X-rays
        image = Image.open(img_path).convert("L")
        mask  = Image.open(mask_path).convert("L")

        # Binarise mask: 0 = background, 1 = lesion/lung region
        mask_np = np.array(mask)
        mask_np = (mask_np > 127).astype(np.uint8)
        mask    = Image.fromarray(mask_np)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask


# ─────────────────────────────────────────────
# 3.  DataLoader Factory
# ─────────────────────────────────────────────
def get_dataloaders(image_dir, mask_dir,
                    image_size=256, batch_size=8,
                    val_split=0.1, test_split=0.1,
                    num_workers=4):
    """
    Creates train / val / test DataLoaders with an 80/10/10 split.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform = JointTransform(image_size=image_size, augment=True)
    eval_transform  = JointTransform(image_size=image_size, augment=False)

    # Full dataset with augmentation ON (we'll override for val/test below)
    full_dataset = ChestXrayDataset(image_dir, mask_dir, transform=train_transform)

    n     = len(full_dataset)
    n_val = int(n * val_split)
    n_tst = int(n * test_split)
    n_trn = n - n_val - n_tst

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_trn, n_val, n_tst],
        generator=torch.Generator().manual_seed(42)
    )

    # Override transforms so val/test don't get augmented
    val_ds.dataset  = ChestXrayDataset(image_dir, mask_dir, transform=eval_transform)
    test_ds.dataset = ChestXrayDataset(image_dir, mask_dir, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"[DataLoader] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# 4.  Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # This will fail if you don't have the data yet — that's expected.
    # Run after downloading the NIH dataset.
    transform = JointTransform(image_size=256, augment=True)
    print("JointTransform created successfully.")

    # Demo with a dummy tensor
    dummy_img  = Image.fromarray(np.random.randint(0, 255, (512, 512), dtype=np.uint8))
    dummy_mask = Image.fromarray(np.random.randint(0,   1, (512, 512), dtype=np.uint8))
    img_t, mask_t = transform(dummy_img, dummy_mask)
    print(f"Image tensor : {img_t.shape}, dtype={img_t.dtype}")
    print(f"Mask tensor  : {mask_t.shape}, dtype={mask_t.dtype}")
