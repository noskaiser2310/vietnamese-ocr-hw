import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


TARGET_HEIGHT = 128
MIN_WIDTH = 128
MAX_WIDTH = 2048
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _composite_on_white(image):
    if image.mode in ("RGBA", "LA"):
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image.convert("RGBA"))
    return image.convert("RGB")


def crop_to_content(image, threshold=245, padding=8):
    """Crop near-white margins while preserving small Vietnamese diacritics."""
    rgb = _composite_on_white(image)
    arr = np.asarray(rgb)
    ink_mask = np.any(arr < threshold, axis=2)

    if not ink_mask.any():
        return rgb

    ys, xs = np.where(ink_mask)
    left = max(int(xs.min()) - padding, 0)
    top = max(int(ys.min()) - padding, 0)
    right = min(int(xs.max()) + padding + 1, rgb.width)
    bottom = min(int(ys.max()) + padding + 1, rgb.height)
    return rgb.crop((left, top, right, bottom))


def resize_and_pad(
    image,
    target_height=TARGET_HEIGHT,
    target_width=None,
    min_width=MIN_WIDTH,
    max_width=MAX_WIDTH,
    crop=True,
):
    """
    Normalize handwriting images for OCR inference/training.

    Keeps aspect ratio, fits the image into a white canvas, and optionally pads
    to a fixed width so very different input shapes reach the model consistently.
    """
    image = crop_to_content(image) if crop else _composite_on_white(image)

    w, h = image.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (target_width or min_width, target_height), "white")

    if target_width is None:
        new_h = target_height
        new_w = int(round(new_h * w / h))
        new_w = max(min_width, min(new_w, max_width))
    else:
        scale = min(target_width / w, target_height / h)
        new_w = max(1, min(int(round(w * scale)), target_width))
        new_h = max(1, min(int(round(h * scale)), target_height))

    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas_w = target_width or new_w
    canvas = Image.new("RGB", (canvas_w, target_height), "white")
    y = max((target_height - new_h) // 2, 0)
    canvas.paste(resized, (0, y))
    return canvas


def to_normalized_tensor(image):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(image)


def preprocess_for_inference(image):
    return to_normalized_tensor(resize_and_pad(image, target_width=MAX_WIDTH))


def preprocess_for_training(image):
    return resize_and_pad(image, target_width=None)
