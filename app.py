"""
Streamlit UI for U-Net Medical Image Segmentation
Auto-downloads model from Google Drive on startup.
Upload a chest X-ray -> see predicted lung segmentation mask.
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import io
import os
import gdown

st.set_page_config(
    page_title="Lung Segmentation AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.stApp{background:#0a0a14;color:#e2e8f0}
#MainMenu,footer,header{visibility:hidden}
.custom-header{
    background:linear-gradient(135deg,#1a0533 0%,#0a0a14 100%);
    border-bottom:1px solid #2a2a3d;
    padding:20px 32px;
    margin:-1rem -1rem 2rem -1rem;
    display:flex;align-items:center;gap:16px;
}
.header-icon{font-size:36px;background:#7c3aed;padding:8px 14px;border-radius:12px}
.header-title{font-size:24px;font-weight:800;
    background:linear-gradient(135deg,#fff 30%,#a78bfa);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.header-sub{font-size:13px;color:#6366f1;margin-top:2px}
.metric-row{display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap}
.metric-card{background:#13131f;border:1px solid #2a2a3d;border-radius:12px;
    padding:16px 20px;flex:1;min-width:120px;text-align:center}
.metric-val{font-size:28px;font-weight:800;color:#a78bfa}
.metric-label{font-size:12px;color:#64748b;margin-top:4px}
.result-title{font-size:13px;font-weight:700;color:#64748b;
    text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;text-align:center}
.dice-badge{background:linear-gradient(135deg,#1a0533,#2d1060);
    border:1px solid #a78bfa;border-radius:12px;padding:20px;text-align:center;margin-top:16px}
.dice-val{font-size:48px;font-weight:800;color:#a78bfa}
.dice-label{font-size:14px;color:#94a3b8;margin-top:4px}
.info-box{background:#0f172a;border-left:3px solid #a78bfa;
    border-radius:0 8px 8px 0;padding:12px 16px;font-size:13px;
    color:#94a3b8;margin-top:12px;line-height:1.6}
.stButton>button{
    background:linear-gradient(135deg,#7c3aed,#a78bfa) !important;
    color:white !important;border:none !important;border-radius:10px !important;
    padding:12px 32px !important;font-size:16px !important;
    font-weight:700 !important;width:100% !important}
section[data-testid="stSidebar"]{
    background:#0d0d1a !important;border-right:1px solid #2a2a3d !important}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# YOUR GOOGLE DRIVE FILE ID — already set!
# ─────────────────────────────────────────────
GDRIVE_FILE_ID = "1TCGKsYmhlv5_OgT3b-OqihASdANmrpUW"
MODEL_PATH     = "outputs/checkpoints/best_model.pth"


# ─────────────────────────────────────────────
# Auto-download model from Google Drive
# ─────────────────────────────────────────────
def download_model_from_drive():
    """Download model from Google Drive if not already present."""
    if os.path.exists(MODEL_PATH):
        return True
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        with st.spinner("⬇️ Downloading model from Google Drive (first time only ~120MB)..."):
            gdown.download(url, MODEL_PATH, quiet=False)
        if os.path.exists(MODEL_PATH):
            return True
        else:
            st.error("Download failed — file not found after download.")
            return False
    except Exception as e:
        st.error(f"❌ Could not download model: {e}")
        st.info("Make sure the Google Drive file is shared as 'Anyone with the link'.")
        return False


# ─────────────────────────────────────────────
# U-Net Architecture
# Key names match trained model exactly
# ─────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        skip = self.conv(x)
        return skip, self.pool(skip)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # named 'upsample' to match trained model keys
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv     = DoubleConv(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoders   = nn.ModuleList()
        prev = in_channels
        for f in features:
            self.encoders.append(EncoderBlock(prev, f))
            prev = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.decoders   = nn.ModuleList()
        prev = features[-1] * 2
        for f in reversed(features):
            self.decoders.append(DecoderBlock(prev, f))
            prev = f
        # named 'final_conv' to match trained model keys
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            skip, x = enc(x)
            skips.append(skip)
        x = self.bottleneck(x)
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)
        return self.final_conv(x)


# ─────────────────────────────────────────────
# Load model (cached — loads only once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    model = UNet(in_channels=1, out_channels=1)
    ckpt  = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


def preprocess(image, size=256):
    img = image.convert("L").resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def run_inference(model, tensor, threshold=0.5):
    probs = torch.sigmoid(model(tensor)).squeeze().numpy()
    mask  = (probs > threshold).astype(np.uint8)
    return probs, mask


def make_overlay(orig_np, mask, alpha=0.4):
    rgb     = np.stack([orig_np] * 3, axis=-1)
    overlay = rgb.copy()
    overlay[mask == 1, 0] = 0
    overlay[mask == 1, 1] = 200
    overlay[mask == 1, 2] = 100
    return (alpha * overlay + (1 - alpha) * rgb).astype(np.uint8)


def dice_score(pred, target):
    inter = (pred * target).sum()
    return (2.0 * inter + 1e-6) / (pred.sum() + target.sum() + 1e-6)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="custom-header">
  <div class="header-icon">🫁</div>
  <div>
    <div class="header-title">Lung Segmentation AI</div>
    <div class="header-sub">U-Net · PyTorch · Built by Shashikant Nikam</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    threshold  = st.slider("Segmentation threshold", 0.1, 0.9, 0.5, 0.05,
                           help="Higher = more conservative segmentation")
    image_size = st.selectbox("Input image size", [128, 256, 512], index=1)
    st.markdown("---")
    st.markdown("### 📖 How it works")
    st.markdown("""
    1. Upload a chest X-ray (PNG/JPG)
    2. U-Net segments the lung regions
    3. See mask + overlay result
    4. Download the segmentation mask
    """)
    st.markdown("---")
    st.markdown("### 🏗️ Architecture")
    st.markdown("""
    - **Encoder**: 4 levels, 64→512 channels
    - **Bottleneck**: 1024 channels
    - **Decoder**: Skip connections
    - **Parameters**: ~31M
    - **Loss**: BCE + Dice combined
    - **Best Dice**: 0.87
    """)
    st.markdown("---")
    st.markdown("### 📊 Dataset")
    st.markdown("""
    - Montgomery County Chest X-ray
    - 138 PA-view X-rays
    - Manual lung field masks
    """)


# ─────────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────────
st.markdown("""
<div class="metric-row">
  <div class="metric-card">
    <div class="metric-val">0.87</div>
    <div class="metric-label">Dice Score</div>
  </div>
  <div class="metric-card">
    <div class="metric-val">31M</div>
    <div class="metric-label">Parameters</div>
  </div>
  <div class="metric-card">
    <div class="metric-val">U-Net</div>
    <div class="metric-label">Architecture</div>
  </div>
  <div class="metric-card">
    <div class="metric-val">256²</div>
    <div class="metric-label">Input Size</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL (auto-download from Drive)
# ─────────────────────────────────────────────
model            = None
model_downloaded = download_model_from_drive()

if model_downloaded and os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        total = sum(p.numel() for p in model.parameters())
        st.success(f"✅ Model loaded successfully — {total:,} parameters")
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
else:
    st.warning("""
    ⚠️ Model not loaded yet.
    Make sure `best_model.pth` is shared publicly on Google Drive.
    Go to Drive → right click file → Share → Anyone with the link.
    """)


# ─────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────
st.markdown("### 📤 Upload Chest X-Ray")
uploaded  = st.file_uploader(
    "Drop a chest X-ray image here",
    type=["png", "jpg", "jpeg"],
    help="Works best with front-facing (PA view) chest X-rays"
)
mask_file = st.file_uploader(
    "Upload ground truth mask (optional — calculates Dice score)",
    type=["png", "jpg", "jpeg"]
)


# ─────────────────────────────────────────────
# INFERENCE + RESULTS
# ─────────────────────────────────────────────
if uploaded and model:
    image = Image.open(uploaded)
    st.markdown("---")
    st.markdown("### 🔬 Segmentation Results")

    with st.spinner("Running U-Net inference..."):
        tensor        = preprocess(image, size=image_size)
        probs, mask   = run_inference(model, tensor, threshold=threshold)
        orig_np       = np.array(image.convert("L").resize((image_size, image_size)))
        overlay_np    = make_overlay(orig_np, mask)
        prob_colored  = cv2.applyColorMap(
            (probs * 255).astype(np.uint8), cv2.COLORMAP_MAGMA
        )

    # 4 panel display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="result-title">Original X-Ray</div>', unsafe_allow_html=True)
        st.image(orig_np, use_column_width=True, clamp=True)
    with col2:
        st.markdown('<div class="result-title">Predicted Mask</div>', unsafe_allow_html=True)
        st.image(mask * 255, use_column_width=True, clamp=True)
    with col3:
        st.markdown('<div class="result-title">Probability Map</div>', unsafe_allow_html=True)
        st.image(prob_colored, use_column_width=True, channels="BGR")
    with col4:
        st.markdown('<div class="result-title">Overlay</div>', unsafe_allow_html=True)
        st.image(overlay_np, use_column_width=True, clamp=True)

    # Stats
    lung_pixels  = int(mask.sum())
    total_pixels = mask.size
    coverage     = lung_pixels / total_pixels * 100

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Lung Coverage",  f"{coverage:.1f}%")
    c2.metric("Lung Pixels",    f"{lung_pixels:,}")
    c3.metric("Threshold Used", f"{threshold:.2f}")

    # Dice score if GT mask uploaded
    if mask_file:
        gt    = Image.open(mask_file).convert("L").resize((image_size, image_size))
        gt_np = (np.array(gt) > 127).astype(np.float32)
        dice  = dice_score(mask.astype(np.float32), gt_np)
        st.markdown(f"""
        <div class="dice-badge">
          <div class="dice-val">{dice:.3f}</div>
          <div class="dice-label">Dice Coefficient vs Ground Truth</div>
        </div>
        """, unsafe_allow_html=True)

    # Download button
    st.markdown("---")
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    buf      = io.BytesIO()
    mask_img.save(buf, format="PNG")
    st.download_button(
        label     = "⬇️  Download Segmentation Mask",
        data      = buf.getvalue(),
        file_name = "lung_mask.png",
        mime      = "image/png"
    )

    st.markdown("""
    <div class="info-box">
    💡 <b>Tips:</b><br>
    • Adjust the <b>threshold slider</b> in the sidebar — higher values = smaller, more precise mask<br>
    • Upload a <b>ground truth mask</b> to get the exact Dice coefficient<br>
    • Works best with <b>PA-view</b> (front-facing) chest X-rays
    </div>
    """, unsafe_allow_html=True)

elif uploaded and not model:
    st.error("❌ Model not loaded. Check the Google Drive sharing settings and refresh the page.")

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:#64748b">
      <div style="font-size:64px;margin-bottom:16px">🫁</div>
      <div style="font-size:18px;font-weight:700;color:#94a3b8;margin-bottom:8px">
        Upload a chest X-ray to begin
      </div>
      <div style="font-size:14px">
        Supports PNG and JPG formats · Works best with PA-view X-rays
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:32px 0 16px;color:#374151;font-size:12px;
border-top:1px solid #2a2a3d;margin-top:32px">
  Built by <span style="color:#a78bfa;font-weight:700">Shashikant Nikam</span>
  · U-Net · PyTorch · Streamlit · Montgomery County Dataset
</div>
""", unsafe_allow_html=True)
