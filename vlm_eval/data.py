from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg"}


@dataclass(frozen=True)
class EvalCase:
    image: str
    image_path: Path
    expected: dict[str, Any]


def find_images(images_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            result[path.name] = path
    return result
