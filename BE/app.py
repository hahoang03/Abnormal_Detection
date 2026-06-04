import io
import base64
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp


# =========================
# CONFIG
# =========================

MODEL_PATH = "hcunet.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(MODEL_PATH, map_location=device)

if "model_state_dict" not in checkpoint:
    raise ValueError("File .pth không đúng format, cần có key model_state_dict")

TRAIN_SIZE = checkpoint.get("train_size", 384)

DEFAULT_MASK_THRESHOLD = checkpoint.get(
    "best_mask_threshold",
    checkpoint.get("best_threshold", 0.35)
)

USE_HIGH_FREQ = checkpoint.get("use_high_frequency", True)
HIGH_FREQ_KERNEL = checkpoint.get("high_freq_kernel", 5)
HIGH_FREQ_SCALE = checkpoint.get("high_freq_scale", 1.0)

IN_CHANNELS = checkpoint.get(
    "in_channels",
    6 if USE_HIGH_FREQ else 3
)

USE_BOTTLENECK_CBAM = checkpoint.get("use_bottleneck_cbam", True)
USE_DECODER_CBAM = checkpoint.get("use_decoder_cbam", False)

CBAM_REDUCTION = checkpoint.get("cbam_reduction", 16)
CBAM_SPATIAL_KERNEL = checkpoint.get("cbam_spatial_kernel", 7)

MIN_AREA_RATIO = 0.0015
CLOSE_KERNEL = 3


# =========================
# HIGH-FREQUENCY INPUT
# =========================

def make_high_frequency(x, kernel_size=HIGH_FREQ_KERNEL):
    pad = kernel_size // 2

    x_pad = F.pad(
        x,
        pad=(pad, pad, pad, pad),
        mode="reflect"
    )

    blur = F.avg_pool2d(
        x_pad,
        kernel_size=kernel_size,
        stride=1,
        padding=0
    )

    high = (x - blur) * HIGH_FREQ_SCALE
    return high


def make_model_input(img):
    if USE_HIGH_FREQ:
        high = make_high_frequency(img)
        return torch.cat([img, high], dim=1)

    return img


# =========================
# CBAM MODULE
# =========================

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()

        hidden = max(in_channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(attn))

        return x * attn


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, spatial_kernel=7):
        super().__init__()

        self.channel_attention = ChannelAttention(
            in_channels=in_channels,
            reduction=reduction
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=spatial_kernel
        )

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# =========================
# MODEL
# =========================

class UnetPlusPlusHighFreqCBAMSegOnly(nn.Module):
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=6,
        classes=1,
        use_bottleneck_cbam=True,
        use_decoder_cbam=False
    ):
        super().__init__()

        self.base = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
            aux_params=None
        )

        bottleneck_channels = self.base.encoder.out_channels[-1]

        if use_bottleneck_cbam:
            self.bottleneck_cbam = CBAM(
                bottleneck_channels,
                reduction=CBAM_REDUCTION,
                spatial_kernel=CBAM_SPATIAL_KERNEL
            )
        else:
            self.bottleneck_cbam = nn.Identity()

        decoder_out_channels = self.base.segmentation_head[0].in_channels

        if use_decoder_cbam:
            self.decoder_cbam = CBAM(
                decoder_out_channels,
                reduction=CBAM_REDUCTION,
                spatial_kernel=CBAM_SPATIAL_KERNEL
            )
        else:
            self.decoder_cbam = nn.Identity()

    def run_decoder(self, features):
        try:
            return self.base.decoder(features)
        except TypeError:
            return self.base.decoder(*features)

    def forward(self, x):
        features = self.base.encoder(x)
        features[-1] = self.bottleneck_cbam(features[-1])

        decoder_output = self.run_decoder(features)
        decoder_output = self.decoder_cbam(decoder_output)

        mask_logits = self.base.segmentation_head(decoder_output)
        return mask_logits


# =========================
# LOAD MODEL ONE TIME
# =========================

model = UnetPlusPlusHighFreqCBAMSegOnly(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=IN_CHANNELS,
    classes=1,
    use_bottleneck_cbam=USE_BOTTLENECK_CBAM,
    use_decoder_cbam=USE_DECODER_CBAM
).to(device)

state_dict = checkpoint["model_state_dict"]
state_dict = {
    k.replace("module.", ""): v
    for k, v in state_dict.items()
}

model.load_state_dict(state_dict, strict=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((TRAIN_SIZE, TRAIN_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# =========================
# HELPER FUNCTIONS
# =========================

def forward_model(model, img):
    x = make_model_input(img)
    return model(x)


def pil_to_base64_png(img_pil):
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def make_overlay(img_pil, mask_pil, alpha=0.45):
    img_np = np.array(img_pil).astype(np.float32)
    mask_np = np.array(mask_pil) > 0

    overlay = img_np.copy()

    red = np.zeros_like(img_np)
    red[:, :, 0] = 255

    overlay[mask_np] = img_np[mask_np] * (1 - alpha) + red[mask_np] * alpha

    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def clean_prediction_mask(mask_pil, min_area_ratio=0.0015, close_kernel=3):
    mask = np.array(mask_pil)
    binary = (mask > 0).astype(np.uint8)

    h, w = binary.shape
    min_area = int(h * w * min_area_ratio)

    kernel = np.ones((close_kernel, close_kernel), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    cleaned = np.zeros_like(binary)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 1

    return Image.fromarray((cleaned * 255).astype(np.uint8))


def predict_one_image(img_pil, threshold, use_postprocess):
    img_pil = img_pil.convert("RGB")
    orig_w, orig_h = img_pil.size

    x = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        mask_logits = forward_model(model, x)
        prob_small = torch.sigmoid(mask_logits)[0, 0].detach().cpu().numpy()

    prob_orig = cv2.resize(
        prob_small,
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR
    )

    raw_prob_min = float(prob_orig.min())
    raw_prob_mean = float(prob_orig.mean())
    raw_prob_max = float(prob_orig.max())

    pred = (prob_orig > threshold).astype(np.uint8) * 255

    prob_img = Image.fromarray(
        np.clip(prob_orig * 255, 0, 255).astype(np.uint8)
    )

    mask_img = Image.fromarray(pred)

    if use_postprocess:
        mask_img = clean_prediction_mask(
            mask_img,
            min_area_ratio=MIN_AREA_RATIO,
            close_kernel=CLOSE_KERNEL
        )

    overlay_img = make_overlay(img_pil, mask_img)

    mask_np = np.array(mask_img) > 0
    pred_area_ratio = float(mask_np.mean())

    return {
        "threshold": float(threshold),
        "probability_map": pil_to_base64_png(prob_img),
        "mask": pil_to_base64_png(mask_img),
        "overlay": pil_to_base64_png(overlay_img),
        "raw_prob_min": raw_prob_min,
        "raw_prob_mean": raw_prob_mean,
        "raw_prob_max": raw_prob_max,
        "pred_area_ratio": pred_area_ratio,
        "train_size": TRAIN_SIZE,
        "default_threshold_from_checkpoint": float(DEFAULT_MASK_THRESHOLD),
        "device": device,
    }


# =========================
# FASTAPI APP
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {
        "message": "HCUNet++ backend is running",
        "device": device,
        "train_size": TRAIN_SIZE,
        "default_threshold": float(DEFAULT_MASK_THRESHOLD),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(DEFAULT_MASK_THRESHOLD),
    use_postprocess: bool = Form(False)
):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = predict_one_image(
        img_pil=img,
        threshold=threshold,
        use_postprocess=use_postprocess
    )

    return result