from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image


def load_and_resize_image(path: Path, max_side: int) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        longest = max(width, height)
        if longest > max_side:
            scale = max_side / longest
            image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))))
        return image.copy()


def encode_image_data_url(path: Path, max_side: int) -> str:
    image = load_and_resize_image(path, max_side)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
