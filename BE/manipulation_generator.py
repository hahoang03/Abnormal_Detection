import io
import base64
import random
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageFile

import torch
from diffusers import StableDiffusionInpaintPipeline

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware


ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================
# CONFIG
# ============================

GEN_SIZE = 512

SEED = 42

MAX_RETRY_PER_IMAGE = 8
NUM_INFERENCE_STEPS = 30

USE_SUBTLE_CUE_PROB = 1.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================
# FASTAPI
# ============================

app = FastAPI(title="Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================
# BASIC UTILS
# ============================

def np_to_data_url(image_np):
    if len(image_np.shape) == 2:
        image = Image.fromarray(image_np).convert("L")
    else:
        image = Image.fromarray(image_np).convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def load_image_rgb_from_bytes(content):
    """
    Đọc ảnh upload và giữ nguyên kích thước gốc.
    Không resize ở bước đọc.
    """
    image = Image.open(io.BytesIO(content)).convert("RGB")
    return np.array(image)


def resize_to_work_size(image_np, size=GEN_SIZE, is_mask=False):
    """
    Resize tạm về 512x512 để generate giống lúc train.
    """
    image = Image.fromarray(image_np)

    if is_mask:
        image = image.resize((size, size), resample=Image.NEAREST)
    else:
        image = image.resize((size, size), resample=Image.BICUBIC)

    return np.array(image)


def resize_back_to_original(image_np, orig_w, orig_h, is_mask=False):
    """
    Resize kết quả từ 512x512 về đúng kích thước ảnh upload ban đầu.
    """
    image = Image.fromarray(image_np)

    if is_mask:
        image = image.resize((orig_w, orig_h), resample=Image.NEAREST)
    else:
        image = image.resize((orig_w, orig_h), resample=Image.BICUBIC)

    return np.array(image)


def keep_only_mask_region(original_np, edited_np, mask_np):
    """
    Giữ thay đổi chỉ nằm trong mask.
    Ngoài mask khôi phục về ảnh gốc.
    """
    original_np = original_np.astype(np.uint8)
    edited_np = edited_np.astype(np.uint8).copy()

    mask = mask_np > 0
    edited_np[~mask] = original_np[~mask]

    return edited_np


def compute_diff_stats(original_np, edited_np, mask_np):
    original = original_np.astype(np.float32)
    edited = edited_np.astype(np.float32)

    mask = mask_np > 0

    if mask.sum() == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "changed_ratio_5": 0.0,
            "changed_ratio_10": 0.0,
            "changed_ratio_20": 0.0,
        }

    diff_map = np.mean(np.abs(original - edited), axis=2)
    values = diff_map[mask]

    return {
        "mean": float(values.mean()),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "changed_ratio_5": float((values > 5).mean()),
        "changed_ratio_10": float((values > 10).mean()),
        "changed_ratio_20": float((values > 20).mean()),
    }


def pass_diff_filter(stats, edit_group):
    """
    Filter để tránh edit quá nhẹ hoặc lỗi.
    Giữ giống logic dataset train.
    """
    if edit_group == "sd_inpaint":
        return (
            10.0 <= stats["mean"] <= 45.0
            and stats["changed_ratio_10"] >= 0.45
            and stats["changed_ratio_20"] >= 0.10
            and stats["p75"] >= 10.0
            and stats["p90"] >= 15.0
        )

    elif edit_group == "shape_overlay":
        return (
            stats["mean"] >= 12.0
            and stats["changed_ratio_10"] >= 0.55
            and stats["p75"] >= 12.0
        )

    else:
        return True


def make_overlay(image_np, mask_np):
    img = image_np.astype(np.float32).copy()
    mask = mask_np > 0

    overlay_color = np.array([255, 0, 0], dtype=np.float32)
    alpha = 0.45

    img[mask] = img[mask] * (1 - alpha) + overlay_color * alpha

    return np.clip(img, 0, 255).astype(np.uint8)


# ============================
# NORMALIZE OPTIONS FROM FRONTEND
# ============================

def normalize_mode(value):
    value = str(value or "sd_inpaint").strip().lower()
    value = value.replace("-", "_").replace(" ", "_")

    if value in ["sdinpaint", "sd_inpaint"]:
        return "sd_inpaint"

    if value in ["shapeoverlay", "shape_overlay", "overlay"]:
        return "shape_overlay"

    if value == "clean":
        return "clean"

    return "sd_inpaint"


def normalize_mask_size(value):
    value = str(value or "random").strip().lower()
    value = value.replace("-", "_").replace(" ", "_")

    mapping = {
        "small": "small",
        "s": "small",

        "medium": "medium",
        "m": "medium",

        "large": "large",
        "l": "large",

        "random": "random",
        "auto": "random",
        "none": "random",
        "": "random",
    }

    return mapping.get(value, "random")


def normalize_shape(value, edit_group="sd_inpaint"):
    value = str(value or "random").strip().lower()
    value = value.replace("-", "_").replace(" ", "_")

    mapping = {
        "ellipse": "ellipse",
        "oval": "ellipse",

        "circle": "circle",
        "round": "circle",

        "rounded_rect": "rounded_rect",
        "rounded_rectangle": "rounded_rect",
        "roundedrect": "rounded_rect",

        "rect": "rect",
        "rectangle": "rect",

        "irregular": "irregular",
        "freeform": "irregular",

        "triangle": "triangle",

        "random": "random",
        "auto": "random",
        "none": "random",
        "": "random",
    }

    shape = mapping.get(value, "random")

    if edit_group == "sd_inpaint":
        # SD inpaint không dùng rect sắc cạnh.
        # Nếu FE gửi rectangle thì đổi sang rounded_rect để mask vẫn tự nhiên hơn.
        if shape == "rect":
            return "rounded_rect"

        if shape not in ["ellipse", "circle", "rounded_rect", "irregular", "random"]:
            return "random"

        return shape

    if edit_group == "shape_overlay":
        if shape not in ["ellipse", "circle", "rect", "rounded_rect", "triangle", "random"]:
            return "random"

        return shape

    return shape


# ============================
# MASK SIZE CONTROL
# ============================

def get_target_ratio():
    level = random.choices(
        ["small", "medium", "large"],
        weights=[0.35, 0.50, 0.15],
        k=1
    )[0]

    if level == "small":
        ratio = random.uniform(0.04, 0.09)
    elif level == "medium":
        ratio = random.uniform(0.09, 0.15)
    else:
        ratio = random.uniform(0.15, 0.20)

    return ratio, level


def get_target_ratio_by_level(requested_level="random"):
    requested_level = normalize_mask_size(requested_level)

    if requested_level == "small":
        return random.uniform(0.04, 0.09), "small"

    if requested_level == "medium":
        return random.uniform(0.09, 0.15), "medium"

    if requested_level == "large":
        return random.uniform(0.15, 0.20), "large"

    return get_target_ratio()


def soften_binary_mask(mask_np, blur_ksize=7, threshold=80):
    """
    Dùng cho inpaint để biên mask đỡ quá răng cưa.
    """
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    blurred = cv2.GaussianBlur(mask_np, (blur_ksize, blur_ksize), 0)
    binary = (blurred > threshold).astype(np.uint8) * 255

    return binary


# ============================
# MASK GENERATION FOR INPAINT
# ============================

def make_ellipse_mask(h, w, target_ratio):
    mask = np.zeros((h, w), np.uint8)

    target_area = target_ratio * h * w
    aspect = random.uniform(0.65, 1.75)

    size_h = int(np.sqrt(target_area / aspect))
    size_w = int(size_h * aspect)

    size_w = int(np.clip(size_w, w // 10, w // 2))
    size_h = int(np.clip(size_h, h // 10, h // 2))

    x1 = random.randint(0, w - size_w)
    y1 = random.randint(0, h - size_h)

    cx = x1 + size_w // 2
    cy = y1 + size_h // 2

    axes = (max(8, size_w // 2), max(8, size_h // 2))
    angle = random.randint(0, 180)

    cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)

    return soften_binary_mask(mask, blur_ksize=7, threshold=80)


def make_circle_mask(h, w, target_ratio):
    mask = np.zeros((h, w), np.uint8)

    target_area = target_ratio * h * w
    radius = int(np.sqrt(target_area / np.pi))
    radius = int(np.clip(radius, min(h, w) // 14, min(h, w) // 4))

    cx = random.randint(radius, w - radius)
    cy = random.randint(radius, h - radius)

    cv2.circle(mask, (cx, cy), radius, 255, -1)

    return soften_binary_mask(mask, blur_ksize=7, threshold=80)


def make_rounded_rect_mask(h, w, target_ratio):
    mask = np.zeros((h, w), np.uint8)

    target_area = target_ratio * h * w
    aspect = random.uniform(0.75, 1.65)

    rect_h = int(np.sqrt(target_area / aspect))
    rect_w = int(rect_h * aspect)

    rect_w = int(np.clip(rect_w, w // 10, w // 2))
    rect_h = int(np.clip(rect_h, h // 10, h // 2))

    x1 = random.randint(0, w - rect_w)
    y1 = random.randint(0, h - rect_h)

    x2 = x1 + rect_w
    y2 = y1 + rect_h

    radius = int(min(rect_w, rect_h) * random.uniform(0.18, 0.35))
    radius = max(8, radius)

    cv2.rectangle(mask, (x1 + radius, y1), (x2 - radius, y2), 255, -1)
    cv2.rectangle(mask, (x1, y1 + radius), (x2, y2 - radius), 255, -1)

    cv2.circle(mask, (x1 + radius, y1 + radius), radius, 255, -1)
    cv2.circle(mask, (x2 - radius, y1 + radius), radius, 255, -1)
    cv2.circle(mask, (x1 + radius, y2 - radius), radius, 255, -1)
    cv2.circle(mask, (x2 - radius, y2 - radius), radius, 255, -1)

    return soften_binary_mask(mask, blur_ksize=7, threshold=80)


def make_irregular_mask(h, w, target_ratio):
    mask = np.zeros((h, w), np.uint8)

    target_area = target_ratio * h * w
    radius = int(np.sqrt(target_area / np.pi))
    radius = int(np.clip(radius, min(h, w) // 14, min(h, w) // 4))

    cx = random.randint(radius, w - radius)
    cy = random.randint(radius, h - radius)

    num_points = random.randint(6, 10)
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    angles += np.random.uniform(-0.25, 0.25, size=num_points)

    points = []

    for angle in angles:
        r = radius * random.uniform(0.65, 1.25)

        x = int(cx + r * np.cos(angle))
        y = int(cy + r * np.sin(angle))

        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))

        points.append([x, y])

    points = np.array(points, np.int32)
    cv2.fillPoly(mask, [points], 255)

    return soften_binary_mask(mask, blur_ksize=7, threshold=80)


def random_inpaint_mask(
    h,
    w,
    max_tries=100,
    requested_level="random",
    requested_shape="random"
):
    """
    Mask cho case SD inpaint.
    Nếu FE gửi shape/level thì dùng đúng shape/level đó.
    Nếu không gửi thì mới random.
    """
    requested_level = normalize_mask_size(requested_level)
    requested_shape = normalize_shape(requested_shape, edit_group="sd_inpaint")

    for _ in range(max_tries):
        target_ratio, level = get_target_ratio_by_level(requested_level)

        if requested_shape == "random":
            shape = random.choices(
                ["ellipse", "circle", "rounded_rect", "irregular"],
                weights=[0.40, 0.20, 0.20, 0.20],
                k=1
            )[0]
        else:
            shape = requested_shape

        if shape == "ellipse":
            mask = make_ellipse_mask(h, w, target_ratio)
        elif shape == "circle":
            mask = make_circle_mask(h, w, target_ratio)
        elif shape == "rounded_rect":
            mask = make_rounded_rect_mask(h, w, target_ratio)
        elif shape == "irregular":
            mask = make_irregular_mask(h, w, target_ratio)
        else:
            mask = make_circle_mask(h, w, target_ratio)
            shape = "circle"

        mask = (mask > 0).astype(np.uint8) * 255
        area_ratio = float((mask > 0).mean())

        if 0.04 <= area_ratio <= 0.20:
            return mask, level, shape, area_ratio

    fallback_shape = requested_shape if requested_shape != "random" else "circle"

    if fallback_shape == "circle":
        mask = make_circle_mask(h, w, 0.10)
    elif fallback_shape == "rounded_rect":
        mask = make_rounded_rect_mask(h, w, 0.10)
    elif fallback_shape == "irregular":
        mask = make_irregular_mask(h, w, 0.10)
    else:
        mask = make_ellipse_mask(h, w, 0.10)
        fallback_shape = "ellipse"

    area_ratio = float((mask > 0).mean())
    fallback_level = requested_level if requested_level != "random" else "medium"

    return mask, fallback_level, fallback_shape, area_ratio


# ============================
# MASK GENERATION FOR SHAPE OVERLAY
# ============================

def make_crisp_circle_mask(h, w, target_ratio):
    mask = np.zeros((h, w), np.uint8)

    target_area = target_ratio * h * w
    radius = int(np.sqrt(target_area / np.pi))
    radius = int(np.clip(radius, min(h, w) // 14, min(h, w) // 4))

    cx = random.randint(radius, w - radius)
    cy = random.randint(radius, h - radius)

    cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask


def make_crisp_ellipse_mask(h, w, target_ratio):
    mask = np.zeros((h, w), np.uint8)

    target_area = target_ratio * h * w
    aspect = random.uniform(0.70, 1.80)

    size_h = int(np.sqrt(target_area / aspect))
    size_w = int(size_h * aspect)

    size_w = int(np.clip(size_w, w // 10, w // 2))
    size_h = int(np.clip(size_h, h // 10, h // 2))

    x1 = random.randint(0, w - size_w)
    y1 = random.randint(0, h - size_h)

    cx = x1 + size_w // 2
    cy = y1 + size_h // 2

    axes = (max(8, size_w // 2), max(8, size_h // 2))
    angle = random.randint(0, 180)

    cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)
    return mask


def make_crisp_rect_mask(h, w, target_ratio):
    mask = np.zeros((h, w), np.uint8)

    target_area = target_ratio * h * w
    aspect = random.uniform(0.70, 1.90)

    rect_h = int(np.sqrt(target_area / aspect))
    rect_w = int(rect_h * aspect)

    rect_w = int(np.clip(rect_w, w // 10, w // 2))
    rect_h = int(np.clip(rect_h, h // 10, h // 2))

    x1 = random.randint(0, w - rect_w)
    y1 = random.randint(0, h - rect_h)

    x2 = x1 + rect_w
    y2 = y1 + rect_h

    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask


def make_crisp_rounded_rect_mask(h, w, target_ratio):
    mask = np.zeros((h, w), np.uint8)

    target_area = target_ratio * h * w
    aspect = random.uniform(0.80, 1.80)

    rect_h = int(np.sqrt(target_area / aspect))
    rect_w = int(rect_h * aspect)

    rect_w = int(np.clip(rect_w, w // 10, w // 2))
    rect_h = int(np.clip(rect_h, h // 10, h // 2))

    x1 = random.randint(0, w - rect_w)
    y1 = random.randint(0, h - rect_h)

    x2 = x1 + rect_w
    y2 = y1 + rect_h

    radius = int(min(rect_w, rect_h) * random.uniform(0.15, 0.30))
    radius = max(8, radius)

    cv2.rectangle(mask, (x1 + radius, y1), (x2 - radius, y2), 255, -1)
    cv2.rectangle(mask, (x1, y1 + radius), (x2, y2 - radius), 255, -1)

    cv2.circle(mask, (x1 + radius, y1 + radius), radius, 255, -1)
    cv2.circle(mask, (x2 - radius, y1 + radius), radius, 255, -1)
    cv2.circle(mask, (x1 + radius, y2 - radius), radius, 255, -1)
    cv2.circle(mask, (x2 - radius, y2 - radius), radius, 255, -1)

    return mask


def make_crisp_triangle_mask(h, w, target_ratio):
    mask = np.zeros((h, w), np.uint8)

    target_area = target_ratio * h * w
    box_side = int(np.sqrt(target_area * 2.0))
    box_side = int(np.clip(box_side, min(h, w) // 8, min(h, w) // 2))

    x1 = random.randint(0, w - box_side)
    y1 = random.randint(0, h - box_side)

    p1 = [x1 + box_side // 2, y1]
    p2 = [x1, y1 + box_side]
    p3 = [x1 + box_side, y1 + box_side]

    pts = np.array([p1, p2, p3], np.int32)
    cv2.fillPoly(mask, [pts], 255)

    return mask


def random_shape_mask(
    h,
    w,
    max_tries=100,
    requested_level="random",
    requested_shape="random"
):
    """
    Mask cho case dán shape kiểu Canva.
    Nếu FE gửi size/shape thì dùng đúng size/shape đó.
    Nếu không gửi thì mới random.
    """
    requested_level = normalize_mask_size(requested_level)
    requested_shape = normalize_shape(requested_shape, edit_group="shape_overlay")

    for _ in range(max_tries):
        target_ratio, level = get_target_ratio_by_level(requested_level)

        if requested_shape == "random":
            shape = random.choices(
                ["rect", "rounded_rect", "circle", "ellipse"],
                weights=[0.35, 0.35, 0.15, 0.15],
                k=1
            )[0]
        else:
            shape = requested_shape

        if shape == "rect":
            mask = make_crisp_rect_mask(h, w, target_ratio)
        elif shape == "rounded_rect":
            mask = make_crisp_rounded_rect_mask(h, w, target_ratio)
        elif shape == "circle":
            mask = make_crisp_circle_mask(h, w, target_ratio)
        elif shape == "ellipse":
            mask = make_crisp_ellipse_mask(h, w, target_ratio)
        elif shape == "triangle":
            mask = make_crisp_triangle_mask(h, w, target_ratio)
        else:
            mask = make_crisp_ellipse_mask(h, w, target_ratio)
            shape = "ellipse"

        mask = (mask > 0).astype(np.uint8) * 255
        area_ratio = float((mask > 0).mean())

        if 0.04 <= area_ratio <= 0.20:
            if requested_level != "random":
                final_level = requested_level
            else:
                if area_ratio < 0.08:
                    final_level = "small"
                elif area_ratio < 0.14:
                    final_level = "medium"
                else:
                    final_level = "large"

            return mask, final_level, shape, area_ratio

    fallback_shape = requested_shape if requested_shape != "random" else "rounded_rect"

    if fallback_shape == "circle":
        mask = make_crisp_circle_mask(h, w, 0.10)
    elif fallback_shape == "ellipse":
        mask = make_crisp_ellipse_mask(h, w, 0.10)
    elif fallback_shape == "rect":
        mask = make_crisp_rect_mask(h, w, 0.10)
    elif fallback_shape == "triangle":
        mask = make_crisp_triangle_mask(h, w, 0.10)
    else:
        mask = make_crisp_rounded_rect_mask(h, w, 0.10)
        fallback_shape = "rounded_rect"

    area_ratio = float((mask > 0).mean())
    fallback_level = requested_level if requested_level != "random" else "medium"

    return mask, fallback_level, fallback_shape, area_ratio


# ============================
# SD INPAINT EDIT
# ============================

def load_sd_inpaint_model():
    sd_device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", sd_device)

    sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if sd_device == "cuda" else torch.float32,
    ).to(sd_device)

    sd_pipe.enable_attention_slicing()

    try:
        sd_pipe.enable_xformers_memory_efficient_attention()
        print("xFormers enabled.")
    except Exception:
        print("xFormers not enabled, continue without it.")

    print("Stable Diffusion Inpaint loaded.")

    return sd_pipe, sd_device


pipe = None
device = None


def get_sd_pipe():
    global pipe, device

    if pipe is None:
        pipe, device = load_sd_inpaint_model()

    return pipe, device


def edit_sd_inpaint(image_pil, mask_np):
    """
    Dùng Stable Diffusion Inpaint trên ảnh 512x512.
    Đây là flow giống lúc tạo dataset train.
    """
    global pipe, device

    get_sd_pipe()

    mask_img = Image.fromarray(mask_np)

    prompt = random.choices(
        [
            "painted patch with slightly different brush strokes",
            "slightly mismatched painted texture",
            "local texture inconsistency in the painting",
            "subtle color inconsistency in painting style",
            "replace the selected area with similar but not identical painted texture",
            "natural continuation of the painting with slight inconsistency",
            "same painting style but locally inconsistent texture"
        ],
        weights=[0.20, 0.20, 0.18, 0.17, 0.12, 0.08, 0.05],
        k=1
    )[0]

    negative_prompt = (
        "text, watermark, logo, frame, border, extra object, face, people, "
        "deformed, low quality, blurry, duplicate object"
    )

    strength = random.uniform(0.68, 0.84)
    guidance_scale = random.uniform(4.5, 6.2)

    sample_generator = torch.Generator(device=device).manual_seed(
        random.randint(0, 10_000_000)
    )

    with torch.inference_mode():
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_pil,
            mask_image=mask_img,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=sample_generator
        )

    if output is None or len(output.images) == 0:
        return np.array(image_pil), prompt, strength, guidance_scale

    return np.array(output.images[0]), prompt, strength, guidance_scale


# ============================
# SUBTLE STATISTICAL CUE FOR SD INPAINT
# ============================

def full_region_alpha(mask_np, erode_iter=2, blur_ksize=31):
    """
    Tạo alpha map:
    - Lõi trong mask = 1.0
    - Biên mask = feather mềm
    - Ngoài mask = 0
    """
    binary = (mask_np > 0).astype(np.uint8)

    if blur_ksize % 2 == 0:
        blur_ksize += 1

    kernel = np.ones((5, 5), np.uint8)

    core = cv2.erode(
        binary,
        kernel,
        iterations=erode_iter
    ).astype(np.float32)

    soft = cv2.GaussianBlur(
        binary.astype(np.float32),
        (blur_ksize, blur_ksize),
        0
    )

    if soft.max() > 0:
        soft = soft / soft.max()

    alpha = np.where(core > 0, 1.0, soft)
    alpha = alpha * binary.astype(np.float32)

    return alpha[..., None]


def apply_subtle_statistical_cue(edited_np, mask_np):
    """
    Cue nhẹ phủ toàn vùng mask:
    - Không thêm noise hạt rời rạc
    - Chỉ chỉnh brightness / contrast / color nhẹ
    - Giúp model có tín hiệu học vùng tampered rõ hơn
    """
    img = edited_np.astype(np.float32)

    alpha_map = full_region_alpha(
        mask_np,
        erode_iter=2,
        blur_ksize=random.choice([31, 41, 51])
    )

    mode = random.choices(
        [
            "region_brightness",
            "region_contrast",
            "region_color_shift",
            "region_brightness_color",
        ],
        weights=[0.30, 0.25, 0.25, 0.20],
        k=1
    )[0]

    modified = img.copy()

    if mode == "region_brightness":
        factor = random.uniform(0.94, 1.06)
        modified = img * factor

    elif mode == "region_contrast":
        mean = img.mean(axis=(0, 1), keepdims=True)
        factor = random.uniform(0.88, 1.12)
        modified = (img - mean) * factor + mean

    elif mode == "region_color_shift":
        shift = np.array([
            random.uniform(-6.0, 6.0),
            random.uniform(-6.0, 6.0),
            random.uniform(-6.0, 6.0)
        ], dtype=np.float32)
        modified = img + shift

    elif mode == "region_brightness_color":
        factor = random.uniform(0.95, 1.05)
        shift = np.array([
            random.uniform(-4.0, 4.0),
            random.uniform(-4.0, 4.0),
            random.uniform(-4.0, 4.0)
        ], dtype=np.float32)
        modified = img * factor + shift

    out = img * (1 - alpha_map) + modified * alpha_map
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out, mode


# ============================
# SHAPE OVERLAY EDIT
# ============================

def make_shape_alpha(mask_np):
    """
    Alpha cho shape overlay.
    Canva thường có biên gọn, có thể hơi anti-alias nhẹ.
    """
    binary = (mask_np > 0).astype(np.float32)

    blur_ksize = random.choice([0, 3, 5, 7])
    opacity = random.uniform(0.72, 1.00)

    if blur_ksize > 0:
        if blur_ksize % 2 == 0:
            blur_ksize += 1

        alpha = cv2.GaussianBlur(binary, (blur_ksize, blur_ksize), 0)

        if alpha.max() > 0:
            alpha = alpha / alpha.max()

        alpha = alpha * binary
    else:
        alpha = binary

    alpha = alpha * opacity

    return alpha[..., None].astype(np.float32), opacity, blur_ksize


def sample_overlay_color():
    """
    Màu shape phổ biến kiểu Canva.
    """
    palette = [
        [255, 59, 48],
        [255, 45, 85],
        [255, 149, 0],
        [255, 204, 0],
        [52, 199, 89],
        [48, 209, 88],
        [10, 132, 255],
        [64, 156, 255],
        [94, 92, 230],
        [175, 82, 222],
        [255, 255, 255],
        [0, 0, 0],
        [28, 28, 30],
        [245, 245, 245],
    ]

    if random.random() < 0.85:
        color = random.choice(palette)
    else:
        color = [
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ]

    return np.array(color, dtype=np.float32).reshape(1, 1, 3)


def apply_shape_overlay(image_np, mask_np):
    """
    Dán shape màu đặc / bán trong suốt lên ảnh gốc 512x512.
    """
    img = image_np.astype(np.float32)
    color = sample_overlay_color()

    alpha, opacity, blur_ksize = make_shape_alpha(mask_np)

    out = img * (1 - alpha) + color * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out, {
        "cue_mode": "shape_overlay",
        "opacity": float(opacity),
        "edge_blur": int(blur_ksize),
        "overlay_color": color.flatten().astype(int).tolist(),
    }


# ============================
# GENERATE ONE SAMPLE
# ============================

def generate_clean_sample(original_np):
    """
    Clean:
    - output giữ đúng size ảnh gốc
    - mask toàn đen đúng size ảnh gốc
    """
    orig_h, orig_w = original_np.shape[:2]

    edited_full = original_np.copy()
    mask_full = np.zeros((orig_h, orig_w), np.uint8)

    info = {
        "edit_group": "clean",
        "edit_type": "clean",
        "level": "none",
        "shape": "none",
        "area_ratio": 0.0,
        "prompt": "none",
        "cue_mode": "none",
        "strength": 0.0,
        "guidance_scale": 0.0,
        "retry_count": 0,
        "status": "success",
        "cue_extra": {},
    }

    return edited_full, mask_full, info


def generate_sd_inpaint_sample(
    original_np,
    requested_level="random",
    requested_shape="random"
):
    """
    SD inpaint:
    - Resize ảnh gốc về 512x512 để generate giống lúc train
    - Tạo mask và inpaint trên 512x512
    - Nếu pass diff filter thì resize edited + mask về size gốc
    - Ngoài mask khôi phục y chang ảnh gốc
    """
    orig_h, orig_w = original_np.shape[:2]

    requested_level = normalize_mask_size(requested_level)
    requested_shape = normalize_shape(requested_shape, edit_group="sd_inpaint")

    work_np = resize_to_work_size(
        original_np,
        size=GEN_SIZE,
        is_mask=False
    )

    image_pil = Image.fromarray(work_np)
    last_info = None

    for retry in range(1, MAX_RETRY_PER_IMAGE + 1):
        mask_small, level, shape, area_ratio = random_inpaint_mask(
            GEN_SIZE,
            GEN_SIZE,
            requested_level=requested_level,
            requested_shape=requested_shape
        )

        edited_small, prompt, strength, guidance_scale = edit_sd_inpaint(
            image_pil=image_pil,
            mask_np=mask_small
        )

        edited_small = keep_only_mask_region(
            original_np=work_np,
            edited_np=edited_small,
            mask_np=mask_small
        )

        if random.random() < USE_SUBTLE_CUE_PROB:
            edited_small, cue_mode = apply_subtle_statistical_cue(
                edited_np=edited_small,
                mask_np=mask_small
            )
        else:
            cue_mode = "none"

        edited_small = keep_only_mask_region(
            original_np=work_np,
            edited_np=edited_small,
            mask_np=mask_small
        )

        stats_small = compute_diff_stats(
            original_np=work_np,
            edited_np=edited_small,
            mask_np=mask_small
        )

        last_info = {
            "edit_group": "sd_inpaint",
            "edit_type": "sd_inpaint",
            "requested_level": requested_level,
            "requested_shape": requested_shape,
            "level": level,
            "shape": shape,
            "area_ratio": area_ratio,
            "prompt": prompt,
            "cue_mode": cue_mode,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "retry_count": retry,
            "status": "last_try",
            "cue_extra": {
                "statistical_cue": cue_mode,
                "work_size": GEN_SIZE,
                "output_size": [orig_w, orig_h],
            },
        }

        if pass_diff_filter(stats_small, "sd_inpaint"):
            last_info["status"] = "success"

            edited_full = resize_back_to_original(
                edited_small,
                orig_w=orig_w,
                orig_h=orig_h,
                is_mask=False
            )

            mask_full = resize_back_to_original(
                mask_small,
                orig_w=orig_w,
                orig_h=orig_h,
                is_mask=True
            )

            mask_full = (mask_full > 0).astype(np.uint8) * 255

            edited_full = keep_only_mask_region(
                original_np=original_np,
                edited_np=edited_full,
                mask_np=mask_full
            )

            return edited_full, mask_full, last_info

    if last_info is None:
        last_info = {
            "edit_group": "sd_inpaint",
            "edit_type": "sd_inpaint",
            "requested_level": requested_level,
            "requested_shape": requested_shape,
            "level": "unknown",
            "shape": "unknown",
            "area_ratio": 0.0,
            "prompt": "none",
            "cue_mode": "none",
            "strength": 0.0,
            "guidance_scale": 0.0,
            "retry_count": 0,
            "status": "failed_unknown",
            "cue_extra": {},
        }

    last_info["status"] = "failed_diff_filter"
    return None, None, last_info


def generate_shape_overlay_sample(
    original_np,
    requested_level="random",
    requested_shape="random"
):
    """
    Shape overlay:
    - Resize ảnh gốc về 512x512 để generate giống lúc train
    - Tạo shape overlay trên 512x512
    - Resize edited + mask về size gốc
    - Ngoài mask khôi phục y chang ảnh gốc
    """
    orig_h, orig_w = original_np.shape[:2]

    requested_level = normalize_mask_size(requested_level)
    requested_shape = normalize_shape(requested_shape, edit_group="shape_overlay")

    work_np = resize_to_work_size(
        original_np,
        size=GEN_SIZE,
        is_mask=False
    )

    last_info = None

    for retry in range(1, MAX_RETRY_PER_IMAGE + 1):
        mask_small, level, shape, area_ratio = random_shape_mask(
            GEN_SIZE,
            GEN_SIZE,
            requested_level=requested_level,
            requested_shape=requested_shape
        )

        edited_small, cue_info = apply_shape_overlay(
            image_np=work_np,
            mask_np=mask_small
        )

        edited_small = keep_only_mask_region(
            original_np=work_np,
            edited_np=edited_small,
            mask_np=mask_small
        )

        stats_small = compute_diff_stats(
            original_np=work_np,
            edited_np=edited_small,
            mask_np=mask_small
        )

        last_info = {
            "edit_group": "shape_overlay",
            "edit_type": "shape_overlay",
            "requested_level": requested_level,
            "requested_shape": requested_shape,
            "level": level,
            "shape": shape,
            "area_ratio": area_ratio,
            "prompt": "none",
            "cue_mode": cue_info.get("cue_mode", "shape_overlay"),
            "strength": 0.0,
            "guidance_scale": 0.0,
            "retry_count": retry,
            "status": "last_try",
            "cue_extra": {
                **cue_info,
                "work_size": GEN_SIZE,
                "output_size": [orig_w, orig_h],
            },
        }

        if pass_diff_filter(stats_small, "shape_overlay"):
            last_info["status"] = "success"

            edited_full = resize_back_to_original(
                edited_small,
                orig_w=orig_w,
                orig_h=orig_h,
                is_mask=False
            )

            mask_full = resize_back_to_original(
                mask_small,
                orig_w=orig_w,
                orig_h=orig_h,
                is_mask=True
            )

            mask_full = (mask_full > 0).astype(np.uint8) * 255

            edited_full = keep_only_mask_region(
                original_np=original_np,
                edited_np=edited_full,
                mask_np=mask_full
            )

            return edited_full, mask_full, last_info

    if last_info is None:
        last_info = {
            "edit_group": "shape_overlay",
            "edit_type": "shape_overlay",
            "requested_level": requested_level,
            "requested_shape": requested_shape,
            "level": "unknown",
            "shape": "unknown",
            "area_ratio": 0.0,
            "prompt": "none",
            "cue_mode": "shape_overlay",
            "strength": 0.0,
            "guidance_scale": 0.0,
            "retry_count": 0,
            "status": "failed_unknown",
            "cue_extra": {},
        }

    last_info["status"] = "failed_diff_filter"
    return None, None, last_info


# ============================
# API ENDPOINT
# ============================

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Colab generator API is running",
        "modes": ["clean", "sd_inpaint", "shape_overlay"],
        "work_size": GEN_SIZE,
        "note": "Generation is done at 512x512 like training, then resized back to original upload size.",
    }


@app.post("/api/generate")
async def generate_image(
    file: UploadFile = File(...),
    mode: str = Form("sd_inpaint"),

    # Nhận kiểu snake_case từ FE
    mask_size: str = Form("random"),
    shape: str = Form("random"),

    # Nhận thêm kiểu camelCase nếu FE đang gửi dạng này
    maskSize: Optional[str] = Form(None),
    inpaint_shape: Optional[str] = Form(None),
    inpaintShape: Optional[str] = Form(None),
):
    content = await file.read()
    original_np = load_image_rgb_from_bytes(content)

    orig_h, orig_w = original_np.shape[:2]

    mode = normalize_mode(mode)

    requested_level = normalize_mask_size(
        maskSize if maskSize is not None else mask_size
    )

    raw_shape = shape

    if inpaint_shape is not None:
        raw_shape = inpaint_shape

    if inpaintShape is not None:
        raw_shape = inpaintShape

    if mode == "shape_overlay":
        requested_shape = normalize_shape(raw_shape, edit_group="shape_overlay")
    else:
        requested_shape = normalize_shape(raw_shape, edit_group="sd_inpaint")

    if mode == "clean":
        edited_np, mask_np, info = generate_clean_sample(original_np)

    elif mode == "sd_inpaint":
        edited_np, mask_np, info = generate_sd_inpaint_sample(
            original_np,
            requested_level=requested_level,
            requested_shape=requested_shape
        )

    elif mode == "shape_overlay":
        edited_np, mask_np, info = generate_shape_overlay_sample(
            original_np,
            requested_level=requested_level,
            requested_shape=requested_shape
        )

    else:
        return {
            "error": f"Invalid mode: {mode}",
            "valid_modes": ["clean", "sd_inpaint", "shape_overlay"],
        }

    if edited_np is None or mask_np is None:
        return {
            "error": f"Generation failed for mode: {mode}",
            "info": info,
        }

    overlay_np = make_overlay(edited_np, mask_np)

    stats = compute_diff_stats(
        original_np=original_np,
        edited_np=edited_np,
        mask_np=mask_np
    )

    return {
        "original": np_to_data_url(original_np),
        "edited": np_to_data_url(edited_np),
        "mask": np_to_data_url(mask_np),
        "overlay": np_to_data_url(overlay_np),
        "info": {
            "edit_group": info.get("edit_group", "unknown"),
            "edit_type": info.get("edit_type", "unknown"),

            # Option FE gửi xuống sau khi normalize
            "requested_level": requested_level,
            "requested_shape": requested_shape,

            # Kết quả mask thật sự được generate
            "level": info.get("level", "none"),
            "shape": info.get("shape", "none"),
            "area_ratio": info.get("area_ratio", 0.0),

            "prompt": info.get("prompt", "none"),
            "cue_mode": info.get("cue_mode", "none"),
            "strength": info.get("strength", 0.0),
            "guidance_scale": info.get("guidance_scale", 0.0),
            "retry_count": info.get("retry_count", 0),
            "status": info.get("status", "success"),
            "extra_info": str(info.get("cue_extra", {})),

            "original_width": int(orig_w),
            "original_height": int(orig_h),
            "work_size": int(GEN_SIZE),

            "diff_mean": stats["mean"],
            "diff_median": stats["median"],
            "diff_p75": stats["p75"],
            "diff_p90": stats["p90"],
            "changed_ratio_5": stats["changed_ratio_5"],
            "changed_ratio_10": stats["changed_ratio_10"],
            "changed_ratio_20": stats["changed_ratio_20"],
        }
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "message": "API is alive",
        "model_loaded": pipe is not None,
        "device": str(device),
    }


@app.get("/api/warmup")
def warmup_model():
    get_sd_pipe()
    return {
        "status": "ok",
        "message": "Stable Diffusion model warmed up",
        "model_loaded": pipe is not None,
        "device": str(device),
    }
