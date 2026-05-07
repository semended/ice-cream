import csv
import io
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PDF_PATH = PROJECT_ROOT / "data" / "raw" / "fair_prices.pdf"

REFERENCE_DIR = PROJECT_ROOT / "data" / "reference_images"
REFERENCE_CANDIDATES_DIR = PROJECT_ROOT / "data" / "reference_candidates"
EMBEDDED_DIR = REFERENCE_CANDIDATES_DIR / "embedded"
PAGE_RENDER_DIR = REFERENCE_CANDIDATES_DIR / "page_renders"
PAGE_CROPS_DIR = REFERENCE_CANDIDATES_DIR / "page_crops"

MANIFEST_CSV = REFERENCE_CANDIDATES_DIR / "reference_candidates_manifest.csv"
CONTACT_SHEET_PATH = REFERENCE_CANDIDATES_DIR / "contact_sheet.jpg"


# ---- Candidate extraction filters (tweak if needed) ----
MIN_EMBEDDED_WIDTH = 250
MIN_EMBEDDED_HEIGHT = 250
MIN_EMBEDDED_AREA = 250 * 250


# ---- Page rendering ----
RENDER_SCALE = 2.0  # 2.0 => good quality, fast enough


@dataclass(frozen=True)
class RelCrop:
    left_ratio: float
    top_ratio: float
    right_ratio: float
    bottom_ratio: float


# Crop boxes are relative to full rendered page (0..1).
# right_ratio is intentionally conservative to cut off competitor thumbnails on the right.
CROP_CONFIG: dict[str, RelCrop] = {
    "cups": RelCrop(left_ratio=0.02, top_ratio=0.10, right_ratio=0.73, bottom_ratio=0.93),
    "eskimo": RelCrop(left_ratio=0.02, top_ratio=0.10, right_ratio=0.73, bottom_ratio=0.93),
    "lakomka": RelCrop(left_ratio=0.02, top_ratio=0.10, right_ratio=0.72, bottom_ratio=0.93),
    "cone": RelCrop(left_ratio=0.02, top_ratio=0.10, right_ratio=0.72, bottom_ratio=0.93),
    "sandwich": RelCrop(left_ratio=0.02, top_ratio=0.10, right_ratio=0.72, bottom_ratio=0.93),
    "large_formats": RelCrop(left_ratio=0.02, top_ratio=0.10, right_ratio=0.75, bottom_ratio=0.94),
}


CATEGORY_SPECS = [
    {
        "key": "cups",
        "out_name": "ref_cups.jpg",
        "keywords": ["стакан", "cups"],
        "crop_key": "cups",
    },
    {
        "key": "eskimo",
        "out_name": "ref_eskimo.jpg",
        "keywords": ["эскимо", "eskimo"],
        "crop_key": "eskimo",
    },
    {
        "key": "lakomka",
        "out_name": "ref_lakomka.jpg",
        "keywords": ["лакомк", "lakomka"],
        "crop_key": "lakomka",
    },
    {
        "key": "cone",
        "out_name": "ref_cone.jpg",
        "keywords": ["рожок", "cone"],
        "crop_key": "cone",
    },
    {
        "key": "sandwich",
        "out_name": "ref_sandwich.jpg",
        "keywords": ["сэндвич", "sandwich"],
        "crop_key": "sandwich",
    },
    {
        "key": "large_formats",
        "out_name": "ref_large_formats.jpg",
        "keywords": ["ведро", "полено", "брикет", "пакет", "large"],
        "crop_key": "large_formats",
    },
]


def _ensure_dirs() -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDED_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_RENDER_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_CROPS_DIR.mkdir(parents=True, exist_ok=True)


def _pil_from_pixmap(pix: fitz.Pixmap) -> Image.Image:
    mode = "RGB"
    if pix.alpha:
        mode = "RGBA"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if img.mode == "RGBA":
        img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img).convert("RGB")
    else:
        img = img.convert("RGB")
    return img


def extract_embedded_candidates(doc: fitz.Document) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []

    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        images = page.get_images(full=True) or []
        if not images:
            continue

        # de-dup xrefs per page
        seen_xrefs: set[int] = set()
        img_seq = 0
        for img in images:
            xref = int(img[0])
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                base = doc.extract_image(xref)
                image_bytes = base["image"]
                pil_img = Image.open(io.BytesIO(image_bytes))
                pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
            except Exception:  # noqa: BLE001
                continue

            w, h = pil_img.size
            area = w * h
            if w < MIN_EMBEDDED_WIDTH or h < MIN_EMBEDDED_HEIGHT or area < MIN_EMBEDDED_AREA:
                continue

            img_seq += 1
            out_path = EMBEDDED_DIR / f"page_{page_idx+1:03d}_img_{img_seq:03d}.jpg"
            pil_img.save(out_path, quality=95)

            rows.append(
                {
                    "path": out_path.relative_to(PROJECT_ROOT).as_posix(),
                    "page": page_idx + 1,
                    "width": w,
                    "height": h,
                    "area": area,
                }
            )

    return rows


def write_manifest(rows: list[dict[str, str | int]]) -> None:
    with MANIFEST_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "page", "width", "height", "area"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_contact_sheet(manifest_rows: list[dict[str, str | int]]) -> None:
    if not manifest_rows:
        return

    # take largest N to keep it readable
    sorted_rows = sorted(manifest_rows, key=lambda r: int(r["area"]), reverse=True)[:48]
    thumbs: list[Image.Image] = []
    for r in sorted_rows:
        p = PROJECT_ROOT / str(r["path"])
        try:
            img = Image.open(p).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        img.thumbnail((320, 320))
        thumbs.append(img)

    if not thumbs:
        return

    cols = 8
    rows_n = (len(thumbs) + cols - 1) // cols
    cell_w, cell_h = 320, 320
    pad = 8
    sheet_w = cols * cell_w + (cols + 1) * pad
    sheet_h = rows_n * cell_h + (rows_n + 1) * pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (245, 245, 245))

    for idx, t in enumerate(thumbs):
        r = idx // cols
        c = idx % cols
        x = pad + c * (cell_w + pad)
        y = pad + r * (cell_h + pad)
        cell = Image.new("RGB", (cell_w, cell_h), (255, 255, 255))
        cell.paste(t, ((cell_w - t.width) // 2, (cell_h - t.height) // 2))
        sheet.paste(cell, (x, y))

    sheet.save(CONTACT_SHEET_PATH, quality=90)


def page_text_lower(page: fitz.Page) -> str:
    try:
        return (page.get_text("text") or "").lower()
    except Exception:  # noqa: BLE001
        return ""


def pick_pages_by_keywords(doc: fitz.Document, keywords: list[str]) -> list[int]:
    picked: list[int] = []
    for page_idx in range(doc.page_count):
        txt = page_text_lower(doc.load_page(page_idx))
        if any(k in txt for k in keywords):
            picked.append(page_idx)
    return picked


def render_page(doc: fitz.Document, page_idx: int) -> Image.Image:
    page = doc.load_page(page_idx)
    pix = page.get_pixmap(matrix=fitz.Matrix(RENDER_SCALE, RENDER_SCALE), alpha=False)
    img = _pil_from_pixmap(pix)
    return img


def crop_rel(img: Image.Image, crop: RelCrop) -> Image.Image:
    w, h = img.size
    left = int(w * crop.left_ratio)
    top = int(h * crop.top_ratio)
    right = int(w * crop.right_ratio)
    bottom = int(h * crop.bottom_ratio)
    left = max(0, min(left, w - 1))
    top = max(0, min(top, h - 1))
    right = max(left + 1, min(right, w))
    bottom = max(top + 1, min(bottom, h))
    return img.crop((left, top, right, bottom))


def stack_vertical(images: list[Image.Image], max_width: int = 1800, pad: int = 12) -> Image.Image:
    if not images:
        raise ValueError("No images to stack")

    # normalize widths
    resized: list[Image.Image] = []
    for im in images:
        if im.width > max_width:
            new_h = int(im.height * (max_width / im.width))
            resized.append(im.resize((max_width, new_h)))
        else:
            resized.append(im)

    w = max(im.width for im in resized)
    h = pad + sum(im.height + pad for im in resized)
    canvas = Image.new("RGB", (w + 2 * pad, h), (255, 255, 255))

    y = pad
    for im in resized:
        x = pad + (w - im.width) // 2
        canvas.paste(im, (x, y))
        y += im.height + pad

    return canvas


def build_page_crops(doc: fitz.Document) -> None:
    for spec in CATEGORY_SPECS:
        key = spec["key"]
        out_name = spec["out_name"]
        crop = CROP_CONFIG[spec["crop_key"]]

        page_indices = pick_pages_by_keywords(doc, spec["keywords"])
        if not page_indices:
            # It's OK: fair_prices.pdf layout may not have text, user can tweak keywords/crops.
            continue

        crops: list[Image.Image] = []
        out_debug_dir = PAGE_CROPS_DIR / key
        out_debug_dir.mkdir(parents=True, exist_ok=True)

        for page_idx in page_indices:
            img = render_page(doc, page_idx)
            # Save full render (for visual debug)
            full_path = PAGE_RENDER_DIR / f"page_{page_idx+1:03d}.jpg"
            img.save(full_path, quality=90)

            cr = crop_rel(img, crop)
            crop_path = out_debug_dir / f"page_{page_idx+1:03d}_crop.jpg"
            cr.save(crop_path, quality=92)
            crops.append(cr)

        if not crops:
            continue

        stacked = stack_vertical(crops)
        out_path = REFERENCE_DIR / out_name
        stacked.save(out_path, quality=92)
        debug_preview_path = REFERENCE_CANDIDATES_DIR / f"debug_{out_name}"
        stacked.save(debug_preview_path, quality=92)


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    _ensure_dirs()

    doc = fitz.open(PDF_PATH)

    # A) Embedded candidates + manifest + contact sheet
    manifest_rows = extract_embedded_candidates(doc)
    write_manifest(manifest_rows)
    build_contact_sheet(manifest_rows)

    # B) Page-based crops as reference sheets (keyword-based auto page selection)
    build_page_crops(doc)

    print(f"Saved reference images to: {REFERENCE_DIR.as_posix()}")
    print(f"Saved debug candidates to: {REFERENCE_CANDIDATES_DIR.as_posix()}")
    print(f"Manifest: {MANIFEST_CSV.as_posix()}")
    print(f"Contact sheet: {CONTACT_SHEET_PATH.as_posix()}")


if __name__ == "__main__":
    main()

