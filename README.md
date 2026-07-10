# Medical Image Segmentation with U-Net

A U-Net built from scratch in PyTorch for semantic segmentation of chest X-ray images (NIH ChestX-ray14 dataset).

## What it does

Given a chest X-ray, the model outputs a pixel-level mask marking the region of interest (e.g. lung fields or a lesion), rather than just a single classification label. This is achieved with the classic U-Net encoder-decoder architecture with skip connections.

## Project Structure

```
unet_segmentation/
├── models/unet.py        # U-Net architecture (encoder, bottleneck, decoder)
├── data/dataset.py        # Dataset loader + augmentation (flip, rotate, elastic deform, jitter)
├── utils/metrics.py       # Dice Loss, BCE+Dice combined loss, Dice/IoU metrics
├── train.py                # Training loop with early stopping, LR scheduling, checkpointing
├── inference.py            # Run inference + 4-panel visualization (original/GT/pred/overlay)
├── requirements.txt
└── outputs/
    ├── checkpoints/         # best_model.pth saved here during training
    └── logs/                # TensorBoard logs
```

## How it works

1. **Encoder**: 4 levels of (Conv→BN→ReLU)×2 + MaxPool, channels grow 64→128→256→512, learning increasingly abstract features while shrinking spatial size.
2. **Bottleneck**: deepest layer (1024 channels), no pooling.
3. **Decoder**: 4 levels of upsampling, each concatenated with the matching encoder layer's output (skip connection) — this recovers the fine spatial detail lost during downsampling.
4. **Output**: 1x1 conv collapses to a single channel = per-pixel logit, turned into a probability mask via sigmoid.

Loss is a 50/50 blend of Binary Cross-Entropy (pixel-level accuracy) and Dice Loss (handles the class imbalance of small foreground regions on a large background).

## Setup — Step by Step

### 1. Clone and set up environment

```bash
git clone <your-repo-url>
cd unet_segmentation

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Get the dataset

Download the NIH ChestX-ray14 dataset:
- Official: https://nihcc.app.box.com/v/ChestXray-NIHCC
- Kaggle mirror: https://www.kaggle.com/datasets/nih-chest-xrays/data

NIH ChestX-ray14 ships without segmentation masks by default — it's a classification dataset. For real lung-field segmentation masks, use one of these instead (recommended for a working pipeline immediately):
- **Montgomery County / Shenzhen CXR Set** (has lung masks): https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen
- **JSRT + SCR masks**: classic lung segmentation benchmark

Arrange the data like this:

```
data/
├── images/
│   ├── img_001.png
│   ├── img_002.png
│   └── ...
└── masks/
    ├── img_001.png   ← same filename as the image, binary mask
    ├── img_002.png
    └── ...
```

### 3. Sanity-check the model (no data needed)

```bash
python models/unet.py
```
Expected output: input/output shapes match, ~31M parameters printed.

### 4. Train

```bash
python train.py \
  --image_dir data/images \
  --mask_dir data/masks \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4
```

This will:
- Split data 80/10/10 train/val/test
- Train with augmentation, validate each epoch without augmentation
- Save the best model (by validation Dice) to `outputs/checkpoints/best_model.pth`
- Log to `outputs/logs/` (view with `tensorboard --logdir outputs/logs`)
- Stop early if val Dice doesn't improve for 10 epochs (`--patience`)

If you don't have a GPU, it'll fall back to CPU automatically (much slower — reduce `--batch_size` and try a small subset of data first to confirm everything runs).

### 5. Run inference + visualize

```bash
python inference.py \
  --checkpoint outputs/checkpoints/best_model.pth \
  --image data/images/img_001.png \
  --mask data/masks/img_001.png \
  --save outputs/sample_prediction.png
```

This produces a 4-panel figure: original X-ray, ground truth mask, predicted mask, and a color-coded overlay (green = correct, red = false positive, blue = false negative), plus prints Dice/IoU for that sample.

## Recommended workflow if you don't have a GPU

Use Google Colab (free GPU):
1. Upload the project folder to Colab or clone your GitHub repo there
2. `!pip install -r requirements.txt`
3. Upload/mount your dataset (Google Drive works well for this)
4. Run the same `train.py` command in a Colab cell with `!python train.py ...`

## Results to expect

With Montgomery/Shenzhen lung masks and ~50 epochs, Dice scores of 0.90+ are typical for lung field segmentation (an easier task than fine lesion segmentation). For lesion-level segmentation, 0.80–0.87 Dice is a strong, realistic result to report.
