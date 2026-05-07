from pathlib import Path
import io

import fitz  # PyMuPDF
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "data" / "raw" / "KIK_report_v3.pdf"
OUTPUT_DIR = PROJECT_ROOT / "data" / "extracted_images"
DEBUG_DIR = PROJECT_ROOT / "data" / "debug_images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# Страницы 3–11 отчета => индексы 2..10
PAGE_INDICES = list(range(2, 11))

# Fallback crop, если встроенную фотку не удалось извлечь.
# Коэффициенты от размеров всей страницы.
# При необходимости потом чуть подкрутим.
CROP_LEFT = 0.05
CROP_TOP = 0.20
CROP_RIGHT = 0.56
CROP_BOTTOM = 0.73


def save_embedded_largest_image(doc: fitz.Document, page: fitz.Page, point_idx: int) -> bool:
    """
    Пытаемся извлечь самую большую встроенную картинку со страницы.
    Возвращает True, если успешно сохранили point_XX.jpg.
    """
    images = page.get_images(full=True)
    if not images:
        return False

    candidates = []
    seen_xrefs = set()

    for img_num, img in enumerate(images, start=1):
        xref = img[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)

        try:
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image.get("ext", "png")

            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            w, h = pil_img.size
            area = w * h

            debug_path = DEBUG_DIR / f"page_{point_idx:02d}_candidate_{img_num:02d}.{ext}"
            pil_img.save(debug_path)

            candidates.append((area, pil_img, debug_path, w, h))
        except Exception:
            continue

    if not candidates:
        return False

    # Берем самую большую картинку по площади
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_img, debug_path, w, h = candidates[0]

    output_path = OUTPUT_DIR / f"point_{point_idx:02d}.jpg"
    best_img.save(output_path, quality=95)

    print(f"[OK] page {point_idx:02d}: extracted embedded image {w}x{h} -> {output_path}")
    print(f"     best candidate source: {debug_path}")
    return True


def save_fallback_crop(page: fitz.Page, point_idx: int) -> None:
    """
    Если встроенную картинку не нашли, рендерим страницу и кропаем левый блок с фото.
    """
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    w, h = img.size
    left = int(w * CROP_LEFT)
    top = int(h * CROP_TOP)
    right = int(w * CROP_RIGHT)
    bottom = int(h * CROP_BOTTOM)

    cropped = img.crop((left, top, right, bottom))

    debug_full_path = DEBUG_DIR / f"page_{point_idx:02d}_full.jpg"
    debug_crop_path = DEBUG_DIR / f"page_{point_idx:02d}_fallback_crop.jpg"

    img.save(debug_full_path, quality=95)
    cropped.save(debug_crop_path, quality=95)

    output_path = OUTPUT_DIR / f"point_{point_idx:02d}.jpg"
    cropped.save(output_path, quality=95)

    print(f"[FALLBACK] page {point_idx:02d}: crop -> {output_path}")
    print(f"           full page: {debug_full_path}")
    print(f"           crop preview: {debug_crop_path}")


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    # Чистим старые point_*.jpg
    for old_file in OUTPUT_DIR.glob("point_*.jpg"):
        old_file.unlink()

    doc = fitz.open(PDF_PATH)

    for i, page_idx in enumerate(PAGE_INDICES, start=1):
        page = doc.load_page(page_idx)

        ok = save_embedded_largest_image(doc, page, i)
        if not ok:
            save_fallback_crop(page, i)

    print("\nDone.")
    print(f"Extracted images: {OUTPUT_DIR}")
    print(f"Debug images: {DEBUG_DIR}")


if __name__ == "__main__":
    main()