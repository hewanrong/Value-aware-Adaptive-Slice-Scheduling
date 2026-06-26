from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


Image.MAX_IMAGE_PIXELS = None


def open_resized_image(path: str | Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.Resampling.BILINEAR)
        return img.copy()


def open_display_image(path: str | Path, long_side: int = 1600) -> tuple[Image.Image, float, float]:
    with Image.open(path) as img:
        img = img.convert("RGB")
        width, height = img.size
        scale = min(1.0, long_side / max(width, height))
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        if new_size != img.size:
            img = img.resize(new_size, Image.Resampling.BILINEAR)
        return img.copy(), new_size[0] / width, new_size[1] / height


def draw_boxes(image: Image.Image, annotations: list[dict], scale_x: float = 1.0, scale_y: float = 1.0) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for gt in annotations:
        x, y, w, h = gt["bbox"]
        color = (0, 200, 83) if gt.get("category_id") == 1 else (41, 121, 255)
        box = [x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y]
        draw.rectangle(box, outline=color, width=3)
    return out


def draw_slice_grid(image: Image.Image, slices: list[dict]) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for s in slices:
        x, y, w, h = s["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=(255, 109, 0), width=2)
    return out
