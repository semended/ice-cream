import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from requests import exceptions as req_exc
from tqdm import tqdm

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = [
    "qwen/qwen3.6-flash",
    "qwen/qwen3.5-plus-20260420",
    "google/gemini-2.5-flash",
    "openai/gpt-4o-mini",
]

INPUT_DIR = Path("data/real_images")
REFERENCE_DIR = Path("data/reference_images")
GROUND_TRUTH_JSONL = Path("data/ground_truth/manual_ground_truth.jsonl")
OUTPUT_CSV = Path("results/openrouter_eval_results.csv")

PREDICTION_FIELDS = [
    "is_trade_equipment_photo",
    "is_ice_cream_equipment",
    "equipment_is_open_freezer",
    "equipment_is_vertical_fridge",
    "equipment_is_display_freezer",
    "equipment_is_branded",
    "photo_quality_score",
    "photo_crop_is_full",
    "photo_crop_is_partial",
    "analysis_possible_score",
    "kik_present",
    "kik_sku_count",
    "kik_share_percent",
    "fill_level_percent",
    "has_cup",
    "has_eskimo",
    "has_lakomka",
    "has_cone",
    "has_sandwich",
    "has_bucket",
    "has_poleno",
    "has_briquette",
    "has_large_pack",
    "has_posm",
    "has_kik_grouped_block",
    "has_kik_products_outside_block",
    "has_foreign_label",
    "has_non_icecream_products",
    "has_empty_sections",
    "is_kik_mixed_with_competitors",
    "kik_outside_block_severity",
    "status_score",
    "confidence_score",
    "uncertainty_notes",
]

BOOLEAN_FIELDS = [
    "is_trade_equipment_photo",
    "is_ice_cream_equipment",
    "equipment_is_open_freezer",
    "equipment_is_vertical_fridge",
    "equipment_is_display_freezer",
    "equipment_is_branded",
    "photo_crop_is_full",
    "photo_crop_is_partial",
    "kik_present",
    "has_cup",
    "has_eskimo",
    "has_lakomka",
    "has_cone",
    "has_sandwich",
    "has_bucket",
    "has_poleno",
    "has_briquette",
    "has_large_pack",
    "has_posm",
    "has_kik_grouped_block",
    "has_kik_products_outside_block",
    "has_foreign_label",
    "has_non_icecream_products",
    "has_empty_sections",
    "is_kik_mixed_with_competitors",
]

NUMERIC_FIELDS = [
    "photo_quality_score",
    "analysis_possible_score",
    "kik_sku_count",
    "kik_share_percent",
    "fill_level_percent",
    "kik_outside_block_severity",
    "status_score",
    "confidence_score",
]

RESPONSE_FORMAT_MODES = {"json_schema", "json_object", "none"}

PROMPT = """Role:
Ты выполняешь аудит полевого фото торгового оборудования с мороженым для контроля мерчандайзинга бренда "Коровка из Кореновки" / "КИК" / "Ренна".

Inputs:
- Reference images: визуальные примеры продукции КИК/Ренна. Это НЕ проверяемые фото и НЕ часть торгового оборудования.
- Target image: единственное фото, которое нужно анализировать.

Output:
- Верни только JSON по схеме. Никакого текста вне JSON.
- Все основные поля должны быть boolean, integer или null.
- uncertainty_notes — короткий список строк с причинами неуверенности или важными видимыми ограничениями.
- Все поля из JSON_SCHEMA обязательны. Не добавляй дополнительные поля.

Decision rules:
- Анализируй только target image. Reference images используй только как визуальную опору, как выглядит продукция КИК/Ренна.
- Не считай reference images частью торгового оборудования, выкладки, SKU, POSM или нарушений.
- Не выдумывай SKU, категории и нарушения, если они не видны.
- Используй null, если признак не видно или нельзя честно определить.
- Не превращай отсутствие уверенности в false.
- false ставь только когда признак явно отсутствует.
- true ставь только когда признак явно виден.
- Если фото плохое, но частично анализируемое, всё равно заполни те поля, которые реально видны, а не ставь всё null.
- Проценты оценивай грубо, шагом 10%: 0, 10, 20, ..., 100.
- Для count используй целое число.
- kik_sku_count — примерное количество визуально различимых SKU КИК/Ренна, не количество упаковок и не количество facings.
- kik_share_percent — примерная визуальная доля продукции КИК/Ренна среди всего мороженого/товаров внутри оборудования.
- fill_level_percent — общая заполненность оборудования товаром, не только КИК.
- Если КИК отсутствует, категории КИК должны быть false или null в зависимости от видимости.
- Если фото не про торговое оборудование: is_trade_equipment_photo=false, analysis_possible_score=0, confidence_score=0, status_score=2, остальные поля null где нельзя определить.
- Если фото про оборудование, но не про мороженое: is_trade_equipment_photo=true, is_ice_cream_equipment=false, analysis_possible_score=0 или 1, status_score=2.
- If photo is not analyzable, use null for visual business fields and explain why in uncertainty_notes.

Consistency rules:
- If kik_present=true, then kik_sku_count must be greater than 0 and kik_share_percent must be greater than 0.
- If kik_sku_count=0 and kik_share_percent=0, then kik_present must be false.
- If uncertainty_notes says that KIK is not found, then kik_present must be false.
- If kik_present=false, all KIK category fields should be false or null.
- Do not return contradictory JSON.

KIK recognition rules:
- Treat products as KIK/Renna/Коровка из Кореновки only if packaging visually matches the provided reference images or visible brand elements.
- Do not require perfect logo readability: packaging color/design/category similarity can be enough if it matches references.
- Use reference images as visual examples of the product family and categories.
- Do not confuse competitor products with KIK.

Share/count rules:
- kik_sku_count is approximate number of distinct visible KIK SKU, not facings/packages.
- If several packages of same design are visible, count them as 1 SKU.
- kik_share_percent is approximate visual share of KIK among all visible products inside the equipment.
- Use coarse 10% steps: 0, 10, 20, ..., 100.
- Do not use exact-looking arbitrary percentages.

KIK block / mixed placement rules:
- Important business rule: a grouped KIK block means KIK products are placed together in a visually compact area of the freezer.
- Do not mark the layout as correct only because a grouped KIK block exists.
- Also check whether any KIK products are placed outside the main KIK block among competitor products.
- has_kik_grouped_block=true if there is a compact KIK area.
- has_kik_products_outside_block=true if one or more KIK products are visible outside that compact area among non-KIK products.
- kik_outside_block_severity:
  0 = no KIK products outside the block.
  1 = 1-2 isolated KIK products outside the block.
  2 = several KIK products outside the block.
  3 = KIK products are strongly mixed with competitors and the grouped block is broken or unclear.
- is_kik_mixed_with_competitors=true if KIK products are mixed with competitor products or placed outside the main grouped block in a way that violates clean block placement.
- If has_kik_grouped_block=true and has_kik_products_outside_block=false, usually is_kik_mixed_with_competitors should be false.
- If KIK is visibly mixed with competitors, has_kik_grouped_block should usually be false or has_kik_products_outside_block should be true.
- Do not mark has_kik_grouped_block=true just because KIK is present.

Status rules:
- status_score=0 normal only if KIK share is high, photo is analyzable, no major merchandising issues are visible.
- status_score=1 attention if KIK is present but there are issues: low/medium share, missing POSM, no grouped KIK block, KIK products outside the block, mixed with competitors, low fill, missing key categories.
- status_score=2 critical if KIK is absent, share is very low, photo is unusable, or severe issues are visible.
- Do not put status_score=0 when there is no POSM, no grouped KIK block, KIK products outside the block, mixed competitors, or low/medium share.

Fill level rules:
- fill_level_percent is total equipment fill, not KIK fill.
- Estimate empty visible space in the equipment.
- Use coarse 10% steps.
- Do not overestimate fill level if large empty zones/low stacks are visible.

Field definitions:
- is_trade_equipment_photo: true если на target image видно торговое оборудование: ларь, морозильник, холодильник, витрина, полка.
- is_ice_cream_equipment: true если оборудование содержит мороженое или явно предназначено для мороженого.
- equipment_is_open_freezer: true для открытого морозильного ларя сверху.
- equipment_is_vertical_fridge: true для вертикального холодильника/морозильника с дверцей/полками.
- equipment_is_display_freezer: true для витринного/секционного ларя, где товар виден сверху/через стекло/по секциям.
- equipment_is_branded: true если оборудование явно брендировано КИК/Ренна/Коровка из Кореновки.
- photo_quality_score: 2 good, 1 medium, 0 bad.
- photo_crop_is_full: true если оборудование полностью или почти полностью попало в кадр.
- photo_crop_is_partial: true если оборудование заметно обрезано, но часть анализа возможна.
- analysis_possible_score: 2 анализ возможен нормально, 1 частично, 0 невозможен.
- kik_present: true если видна продукция КИК/Ренна/Коровка из Кореновки.
- kik_sku_count: примерное количество визуально различимых SKU КИК/Ренна, не количество упаковок.
- kik_share_percent: примерная визуальная доля КИК/Ренна внутри оборудования, 0..100.
- fill_level_percent: общая заполненность оборудования товаром, 0..100.
- has_cup: видны стаканчики КИК.
- has_eskimo: видно эскимо КИК.
- has_lakomka: видна лакомка КИК.
- has_cone: видны рожки КИК.
- has_sandwich: видны сэндвичи КИК.
- has_bucket: видны ведёрки КИК.
- has_poleno: видно полено КИК.
- has_briquette: видны брикеты КИК.
- has_large_pack: видны большие упаковки/пакеты КИК.
- has_posm: true если видны фирменные POSM, фирменные ценники, брендированные материалы, воблеры или явные брендированные ценники КИК/Ренна.
- has_kik_grouped_block: true если продукция КИК визуально собрана в отдельный чистый блок.
- has_kik_products_outside_block: true если один или несколько продуктов КИК находятся вне основного блока среди конкурентов.
- kik_outside_block_severity: 0 no outside products, 1 one-two isolated products, 2 several outside products, 3 strongly mixed / block broken or unclear.
- has_foreign_label: true если видна чужая бирка/ценник/табличка, не соответствующая мороженому или КИК, например "МОЛОКО", "КОЛБАСА".
- has_non_icecream_products: true если в/на оборудовании видны не мороженые продукты: выпечка, молоко, овощи, полуфабрикаты и т.п.
- has_empty_sections: true если заметны пустые секции/корзины/полки оборудования.
- is_kik_mixed_with_competitors: true если продукция КИК перемешана с конкурентами или размещена вне основного блока так, что нарушает чистую блочную выкладку.
- status_score: 0 normal, 1 attention, 2 critical.
- confidence_score: 2 high, 1 medium, 0 low.
- uncertainty_notes: короткий список причин неуверенности или важных видимых ограничений: блики, обрезка, плохой угол, мелкие упаковки, закрытые ценники, спорная категория.

Business status rules:
- status_score=0 normal: КИК хорошо представлен, фото пригодно, критичных нарушений нет.
- status_score=1 attention: есть нарушения, но КИК присутствует и анализ возможен.
- status_score=2 critical: КИК отсутствует, доля сильно низкая, фото непригодно, либо есть серьёзные нарушения.
- confidence_score=2 high: признаки хорошо видны.
- confidence_score=1 medium: есть ограничения, но анализ в целом возможен.
- confidence_score=0 low: плохое фото / сильные блики / обрезка / мало уверенности.

Output discipline:
Перед финальным JSON внутренне проверь consistency rules, but output only JSON.
Верни только JSON. Все поля из JSON_SCHEMA обязательны. additionalProperties=false. Do not output deprecated fields such as has_monobrand_block.
"""

JSON_SCHEMA: dict[str, Any] = {
    "name": "kik_countable_audit_schema",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "is_trade_equipment_photo": {"type": ["boolean", "null"]},
            "is_ice_cream_equipment": {"type": ["boolean", "null"]},
            "equipment_is_open_freezer": {"type": ["boolean", "null"]},
            "equipment_is_vertical_fridge": {"type": ["boolean", "null"]},
            "equipment_is_display_freezer": {"type": ["boolean", "null"]},
            "equipment_is_branded": {"type": ["boolean", "null"]},
            "photo_quality_score": {"type": ["integer", "null"], "enum": [0, 1, 2, None]},
            "photo_crop_is_full": {"type": ["boolean", "null"]},
            "photo_crop_is_partial": {"type": ["boolean", "null"]},
            "analysis_possible_score": {"type": ["integer", "null"], "enum": [0, 1, 2, None]},
            "kik_present": {"type": ["boolean", "null"]},
            "kik_sku_count": {"type": ["integer", "null"], "minimum": 0},
            "kik_share_percent": {"type": ["integer", "null"], "minimum": 0, "maximum": 100},
            "fill_level_percent": {"type": ["integer", "null"], "minimum": 0, "maximum": 100},
            "has_cup": {"type": ["boolean", "null"]},
            "has_eskimo": {"type": ["boolean", "null"]},
            "has_lakomka": {"type": ["boolean", "null"]},
            "has_cone": {"type": ["boolean", "null"]},
            "has_sandwich": {"type": ["boolean", "null"]},
            "has_bucket": {"type": ["boolean", "null"]},
            "has_poleno": {"type": ["boolean", "null"]},
            "has_briquette": {"type": ["boolean", "null"]},
            "has_large_pack": {"type": ["boolean", "null"]},
            "has_posm": {"type": ["boolean", "null"]},
            "has_kik_grouped_block": {"type": ["boolean", "null"]},
            "has_kik_products_outside_block": {"type": ["boolean", "null"]},
            "has_foreign_label": {"type": ["boolean", "null"]},
            "has_non_icecream_products": {"type": ["boolean", "null"]},
            "has_empty_sections": {"type": ["boolean", "null"]},
            "is_kik_mixed_with_competitors": {"type": ["boolean", "null"]},
            "kik_outside_block_severity": {"type": ["integer", "null"], "enum": [0, 1, 2, 3, None]},
            "status_score": {"type": ["integer", "null"], "enum": [0, 1, 2, None]},
            "confidence_score": {"type": ["integer", "null"], "enum": [0, 1, 2, None]},
            "uncertainty_notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": PREDICTION_FIELDS,
        "additionalProperties": False,
    },
}


def image_to_data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    ext = image_path.suffix.lower().replace(".", "")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    return f"data:image/{mime};base64,{encoded}"


def extract_json_from_response(response_json: dict[str, Any]) -> dict[str, Any]:
    content = response_json["choices"][0]["message"].get("content", "")
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        content = "\n".join(text_parts)
    if isinstance(content, dict):
        return content

    text = str(content).strip()
    # common failure mode: extra text / markdown fences
    if text.startswith("```"):
        text = text.strip("`").strip()
    if "{" in text and "}" in text:
        text = text[text.find("{") : text.rfind("}") + 1]
    return json.loads(text)


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "unknown", "null", "none", "nan", "-"):
        return None
    if text in ("true", "1", "yes", "y", "да", "истина"):
        return True
    if text in ("false", "0", "no", "n", "нет", "ложь"):
        return False
    return None


def parse_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip().lower()
    if text in ("", "unknown", "null", "none", "nan", "-"):
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def parse_severity(value: Any) -> int | None:
    parsed = parse_int(value)
    if parsed is None or parsed < 0 or parsed > 3:
        return None
    return parsed


def normalize_prediction(output: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(output, dict):
        raise ValueError("Parsed prediction JSON must be an object")

    normalized: dict[str, Any] = {}
    for field in PREDICTION_FIELDS:
        value = output.get(field)
        if field == "has_kik_grouped_block" and value is None:
            value = output.get("has_monobrand_block")
        if field in BOOLEAN_FIELDS:
            normalized[field] = parse_bool(value)
        elif field == "kik_outside_block_severity":
            normalized[field] = parse_severity(value)
        elif field in NUMERIC_FIELDS:
            normalized[field] = parse_int(value)
        elif field == "uncertainty_notes":
            if isinstance(value, list):
                normalized[field] = [str(item) for item in value if item is not None]
            elif value in (None, ""):
                normalized[field] = []
            else:
                normalized[field] = [str(value)]
        else:
            normalized[field] = value
    return normalized


def response_format_mode_for_model(model: str) -> str:
    explicit_mode = os.getenv("RESPONSE_FORMAT_MODE", "").strip().lower()
    if explicit_mode:
        if explicit_mode not in RESPONSE_FORMAT_MODES:
            modes = ", ".join(sorted(RESPONSE_FORMAT_MODES))
            raise ValueError(f"RESPONSE_FORMAT_MODE must be one of: {modes}")
        return explicit_mode

    if model.lower().startswith("google/gemini-"):
        return "none"
    return "json_schema"


def response_format_for_mode(mode: str) -> dict[str, Any] | None:
    if mode == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": JSON_SCHEMA,
        }
    if mode == "json_object":
        return {"type": "json_object"}
    if mode == "none":
        return None
    raise ValueError(f"Unsupported response_format mode: {mode}")


def _reference_items() -> list[tuple[str, Path]]:
    if not REFERENCE_DIR.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = sorted([p for p in REFERENCE_DIR.iterdir() if p.suffix.lower() in exts])

    labeled: list[tuple[str, Path]] = []
    for p in paths:
        stem = p.stem.lower()
        if stem.startswith("ref_"):
            key = stem.replace("ref_", "").replace("_", " ")
            label = f"Reference image ({key}):"
        else:
            label = f"Reference image ({p.name}):"
        labeled.append((label, p))
    return labeled


def load_ground_truth_image_ids() -> set[str] | None:
    if not GROUND_TRUTH_JSONL.exists():
        return None

    image_ids: set[str] = set()
    with GROUND_TRUTH_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = str(row.get("image_id", "")).strip()
            if image_id:
                image_ids.add(image_id)
    return image_ids


def selected_models_from_env() -> list[str]:
    raw = os.getenv("OPENROUTER_MODELS", "").strip()
    if not raw:
        return MODELS
    models = [model.strip() for model in raw.split(",") if model.strip()]
    return models or MODELS


def max_images_from_env() -> int:
    raw = os.getenv("MAX_IMAGES", "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError as exc:
        raise ValueError("MAX_IMAGES must be a non-negative integer") from exc


def select_image_paths() -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    all_image_paths = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in exts])
    ground_truth_image_ids = load_ground_truth_image_ids()

    if ground_truth_image_ids is None:
        filtered_paths = all_image_paths
        skipped_paths: list[Path] = []
        ground_truth_count = 0
        print(f"[WARN] Ground truth JSONL not found: {GROUND_TRUTH_JSONL.as_posix()}. Using all images.")
    else:
        filtered_paths = [p for p in all_image_paths if p.name in ground_truth_image_ids]
        skipped_paths = [p for p in all_image_paths if p.name not in ground_truth_image_ids]
        ground_truth_count = len(ground_truth_image_ids)

    max_images = max_images_from_env()
    image_paths = filtered_paths[:max_images] if max_images > 0 else filtered_paths

    print(f"Images found in {INPUT_DIR.as_posix()}: {len(all_image_paths)}")
    print(f"Image IDs found in ground truth: {ground_truth_count}")
    print(f"Images selected for evaluation: {len(image_paths)}")
    if max_images > 0:
        print(f"MAX_IMAGES applied: {max_images}")
    if skipped_paths:
        skipped_names = ", ".join(p.name for p in skipped_paths)
        print(f"Skipped because not in ground truth: {skipped_names}")
    else:
        print("Skipped because not in ground truth: none")

    return image_paths


def call_model(
    api_key: str,
    model: str,
    target_image_path: Path,
    reference_items: list[tuple[str, Path]],
    response_format_mode: str,
) -> tuple[dict[str, Any] | None, str | None, float]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    content: list[dict[str, Any]] = [{"type": "text", "text": PROMPT}]

    for label, ref_path in reference_items:
        content.append({"type": "text", "text": label})
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(ref_path)}})

    content.append({"type": "text", "text": "TARGET IMAGE TO ANALYZE"})
    content.append({"type": "image_url", "image_url": {"url": image_to_data_url(target_image_path)}})

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }
    response_format = response_format_for_mode(response_format_mode)
    if response_format is not None:
        payload["response_format"] = response_format

    last_err: str | None = None
    started = time.time()
    for attempt in range(3):  # first try + 2 retries
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )

            if response.status_code in (408, 429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2.0 + attempt * 2.0)
                continue

            if response.status_code >= 400:
                latency_sec = time.time() - started
                return None, f"HTTP {response.status_code}: {response.text[:1000]}", latency_sec

            parsed = response.json()
            model_output = normalize_prediction(extract_json_from_response(parsed))
            latency_sec = time.time() - started
            return model_output, None, latency_sec
        except (ConnectionResetError, req_exc.Timeout, req_exc.ConnectionError) as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt < 2:
                time.sleep(1.0 + attempt * 1.5)
                continue
            latency_sec = time.time() - started
            return None, last_err, latency_sec
        except Exception as exc:  # noqa: BLE001
            latency_sec = time.time() - started
            return None, f"{type(exc).__name__}: {exc}", latency_sec

    latency_sec = time.time() - started
    return None, last_err or "Unknown error", latency_sec


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set. Create .env file from .env.example")

    image_paths = select_image_paths()
    if not image_paths:
        raise FileNotFoundError(f"No images found in {INPUT_DIR}.")

    selected_models = selected_models_from_env()
    print(f"Models selected: {', '.join(selected_models)}")

    reference_items = _reference_items()
    if not reference_items:
        print(f"[WARN] No reference images found in {REFERENCE_DIR.as_posix()}. Running without references.")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    total = len(selected_models) * len(image_paths)
    with tqdm(total=total, desc="Running VLM eval") as pbar:
        for model in selected_models:
            response_format_mode = response_format_mode_for_model(model)
            print(f"Model: {model}")
            print(f"Selected response_format_mode: {response_format_mode}")
            for image_path in image_paths:
                output, error, latency_sec = call_model(
                    api_key,
                    model,
                    image_path,
                    reference_items,
                    response_format_mode,
                )

                row = {
                    "model": model,
                    "image_id": image_path.name,
                    "latency_sec": round(latency_sec, 3),
                    "error": error or "",
                }

                if output is not None:
                    row["prediction_json"] = json.dumps(output, ensure_ascii=False)
                    for field in PREDICTION_FIELDS:
                        if field == "uncertainty_notes":
                            row[field] = json.dumps(output.get(field, []), ensure_ascii=False)
                        else:
                            row[field] = output.get(field)
                else:
                    row["prediction_json"] = ""
                    for field in PREDICTION_FIELDS:
                        row[field] = ""

                rows.append(row)
                pbar.update(1)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUTPUT_CSV.as_posix()}")


if __name__ == "__main__":
    main()
