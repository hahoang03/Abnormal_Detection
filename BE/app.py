import io
import base64
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp


# =========================
# CONFIG
# =========================

MODEL_PATHS = {
    "hcunetpp": "hcunet.pth",
    "unetpp": "unetplus_baseline.pth",
}

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Device:", device)

MIN_AREA_RATIO = 0.0015
CLOSE_KERNEL = 3


# =========================
# HIGH-FREQUENCY INPUT
# =========================

def make_high_frequency(x, kernel_size=5, scale=1.0):
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

    high = (x - blur) * scale
    return high


def make_model_input(img, model_info):
    if model_info["use_high_frequency"]:
        high = make_high_frequency(
            img,
            kernel_size=model_info["high_freq_kernel"],
            scale=model_info["high_freq_scale"]
        )
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
        use_decoder_cbam=False,
        cbam_reduction=16,
        cbam_spatial_kernel=7
        
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
                reduction=cbam_reduction,
                spatial_kernel=cbam_spatial_kernel
            )
        else:
            self.bottleneck_cbam = nn.Identity()

        decoder_out_channels = self.base.segmentation_head[0].in_channels

        if use_decoder_cbam:
            self.decoder_cbam = CBAM(
                    decoder_out_channels,
                    reduction=cbam_reduction,
                    spatial_kernel=cbam_spatial_kernel
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



#HELPER FUNCTION

def clean_state_dict(state_dict):
    return {
        k.replace("module.", ""): v
        for k, v in state_dict.items()
    }


def state_dict_uses_wrapper(state_dict):
    return any(
        k.startswith("base.")
        or k.startswith("bottleneck_cbam.")
        or k.startswith("decoder_cbam.")
        for k in state_dict.keys()
    )


def load_model_bundle(model_type, model_path):
    checkpoint = torch.load(model_path, map_location="cpu")

    if "model_state_dict" not in checkpoint:
        raise ValueError(f"{model_path} không đúng format, cần có key model_state_dict")

    state_dict = clean_state_dict(checkpoint["model_state_dict"])

    train_size = checkpoint.get("train_size", 384)

    default_threshold = checkpoint.get(
        "best_mask_threshold",
        checkpoint.get("best_threshold", 0.35)
    )

    # HCUNet++ mặc định dùng high-frequency input.
    # U-Net++ baseline mặc định dùng RGB 3 channels.
    default_use_high_frequency = model_type == "hcunetpp"

    use_high_frequency = checkpoint.get(
        "use_high_frequency",
        default_use_high_frequency
    )

    high_freq_kernel = checkpoint.get("high_freq_kernel", 5)
    high_freq_scale = checkpoint.get("high_freq_scale", 1.0)

    in_channels = checkpoint.get(
        "in_channels",
        6 if use_high_frequency else 3
    )

    has_bottleneck_cbam = any(
        k.startswith("bottleneck_cbam.")
        for k in state_dict.keys()
    )

    has_decoder_cbam = any(
        k.startswith("decoder_cbam.")
        for k in state_dict.keys()
    )

    use_bottleneck_cbam = checkpoint.get(
        "use_bottleneck_cbam",
        has_bottleneck_cbam
    )

    use_decoder_cbam = checkpoint.get(
        "use_decoder_cbam",
        has_decoder_cbam
    )

    cbam_reduction = checkpoint.get("cbam_reduction", 16)
    cbam_spatial_kernel = checkpoint.get("cbam_spatial_kernel", 7)

    uses_wrapper = state_dict_uses_wrapper(state_dict)

    if uses_wrapper:
        loaded_model = UnetPlusPlusHighFreqCBAMSegOnly(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=in_channels,
            classes=1,
            use_bottleneck_cbam=use_bottleneck_cbam,
            use_decoder_cbam=use_decoder_cbam,
            cbam_reduction=cbam_reduction,
            cbam_spatial_kernel=cbam_spatial_kernel
        ).to(device)
    else:
        loaded_model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=in_channels,
            classes=1,
            activation=None,
            aux_params=None
        ).to(device)

    loaded_model.load_state_dict(state_dict, strict=True)
    loaded_model.eval()

    model_transform = transforms.Compose([
        transforms.Resize((train_size, train_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    return {
        "model": loaded_model,
        "path": model_path,
        "train_size": train_size,
        "default_threshold": float(default_threshold),
        "use_high_frequency": bool(use_high_frequency),

        # Baseline U-Net++ không dùng high-frequency nên checkpoint lưu None
        "high_freq_kernel": int(high_freq_kernel) if high_freq_kernel is not None else None,
        "high_freq_scale": float(high_freq_scale) if high_freq_scale is not None else None,

        "in_channels": int(in_channels),

        "use_bottleneck_cbam": bool(use_bottleneck_cbam),
        "use_decoder_cbam": bool(use_decoder_cbam),

        # Baseline U-Net++ không dùng CBAM nên checkpoint lưu None
        "cbam_reduction": int(cbam_reduction) if cbam_reduction is not None else None,
        "cbam_spatial_kernel": int(cbam_spatial_kernel) if cbam_spatial_kernel is not None else None,

        "transform": model_transform,
    }


MODEL_BUNDLES = {
    model_type: load_model_bundle(model_type, model_path)
    for model_type, model_path in MODEL_PATHS.items()
}

# =========================
# HELPER FUNCTIONS
# =========================

def forward_model(model_info, img):
    x = make_model_input(img, model_info)
    return model_info["model"](x)


def pil_to_base64_png(img_pil):
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def make_overlay(img_pil, mask_pil, alpha=0.45, overlay_mode="fill"):
    img_np = np.array(img_pil.convert("RGB")).astype(np.uint8)
    mask_np = (np.array(mask_pil.convert("L")) > 0).astype(np.uint8)

    overlay = img_np.copy()

    if overlay_mode == "contour":
        contours, _ = cv2.findContours(
            mask_np,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) > 0:
            # Viền trắng bên ngoài để nổi trên nền tối/sáng
            cv2.drawContours(
                overlay,
                contours,
                -1,
                (255, 255, 255),
                thickness=6,
                lineType=cv2.LINE_AA
            )

            # Viền đỏ chính
            cv2.drawContours(
                overlay,
                contours,
                -1,
                (255, 0, 0),
                thickness=3,
                lineType=cv2.LINE_AA
            )

        return Image.fromarray(overlay)

    # Default: fill red region for inpaint
    img_float = img_np.astype(np.float32)

    red = np.zeros_like(img_float)
    red[:, :, 0] = 255

    mask_bool = mask_np > 0

    img_float[mask_bool] = (
        img_float[mask_bool] * (1 - alpha)
        + red[mask_bool] * alpha
    )

    return Image.fromarray(np.clip(img_float, 0, 255).astype(np.uint8))



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

    
def predict_one_image(
    img_pil,
    threshold,
    use_postprocess,
    overlay_mode="fill",
    model_type="hcunetpp"
):
    model_info = MODEL_BUNDLES[model_type]
    img_pil = img_pil.convert("RGB")
    orig_w, orig_h = img_pil.size

    x = model_info["transform"](img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        mask_logits = forward_model(model_info, x)
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

    overlay_img = make_overlay(
        img_pil,
        mask_img,
        overlay_mode=overlay_mode
    )

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
        "model_type": model_type,
        "model_path": model_info["path"],
        "train_size": model_info["train_size"],
        "default_threshold_from_checkpoint": float(model_info["default_threshold"]),
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
        "message": "Detection backend is running",
        "device": device,
        "available_models": {
            model_type: {
                "path": info["path"],
                "train_size": info["train_size"],
                "default_threshold": info["default_threshold"],
                "in_channels": info["in_channels"],
                "use_high_frequency": info["use_high_frequency"],
                "use_bottleneck_cbam": info["use_bottleneck_cbam"],
                "use_decoder_cbam": info["use_decoder_cbam"],
            }
            for model_type, info in MODEL_BUNDLES.items()
        }
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    use_postprocess: bool = Form(False),
    overlay_mode: str = Form("fill"),
    model_type: str = Form("hcunetpp")
):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    model_type = model_type.lower().strip()

    if model_type not in MODEL_BUNDLES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type: {model_type}. Use one of: {list(MODEL_BUNDLES.keys())}"
        )

    if threshold is None:
        threshold = MODEL_BUNDLES[model_type]["default_threshold"]

    overlay_mode = overlay_mode.lower().strip()

    if overlay_mode not in ["fill", "contour"]:
        overlay_mode = "fill"

    result = predict_one_image(
        img_pil=img,
        threshold=threshold,
        use_postprocess=use_postprocess,
        overlay_mode=overlay_mode,
        model_type=model_type
    )

    return result