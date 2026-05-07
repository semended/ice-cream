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
    "has_monobrand_block",
    "has_foreign_label",
    "has_non_icecream_products",
    "has_empty_sections",
    "is_kik_mixed_with_competitors",
    "status_score",
    "confidence_score",
    "uncertainty_notes",
]

PROMPT = """Ты выполняешь визуальный аудит торгового оборудования с мороженым для бизнес-задачи контроля представленности бренда "Коровка из Кореновки" / "КИК" / "Ренна".

На входе тебе даны:
1. reference images — визуальные примеры продукции КИК/Ренна;
2. target image — полевое фото торгового оборудования, которое нужно проанализировать.

Reference images используй только как визуальную опору для понимания того, как выглядит продукция КИК. Анализировать нужно только target image.

Твоя задача — вернуть строго структурированный JSON с бинарными и числовыми признаками.

Нужно определить по target image:

1. Является ли фото снимком торгового оборудования.
2. Есть ли на фото оборудование с мороженым.
3. Тип оборудования в countable виде:
   - equipment_is_open_freezer
   - equipment_is_vertical_fridge
   - equipment_is_display_freezer
   - equipment_is_branded
4. Есть ли продукция КИК / Коровка из Кореновки / Ренна.
5. Сколько примерно SKU КИК видно.
6. Какую визуальную долю занимает КИК в оборудовании, в процентах.
7. Общую заполненность оборудования товаром, в процентах.
8. Какие категории КИК видны:
   - стакан
   - эскимо
   - лакомка
   - рожок
   - сэндвич
   - ведро
   - полено
   - брикет
   - пакет / крупная упаковка
9. Видимые нарушения:
   - нет POSM / фирменных ценников / воблеров
   - нет монобрендового блока
   - КИК перемешан с конкурентами
   - видны посторонние товары
   - видны пустые секции
   - фото плохого качества
   - фото обрезано

Правила:
- Не выдумывай SKU, категории и нарушения, если они не видны.
- Если не уверен по boolean-полю, ставь null.
- Проценты оценивай грубо, шагом 10%.
- Для photo_quality_score:
  2 = хорошее
  1 = среднее
  0 = плохое
- Для analysis_possible_score:
  2 = можно анализировать нормально
  1 = можно анализировать частично
  0 = анализ невозможен
- Для status_score:
  0 = normal
  1 = attention
  2 = critical
- Для confidence_score:
  2 = high
  1 = medium
  0 = low
- Ответ должен быть только JSON.
- Не добавляй никакой текст вне JSON.
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
            "photo_quality_score": {"type": ["integer", "null"]},
            "photo_crop_is_full": {"type": ["boolean", "null"]},
            "photo_crop_is_partial": {"type": ["boolean", "null"]},
            "analysis_possible_score": {"type": ["integer", "null"]},
            "kik_present": {"type": ["boolean", "null"]},
            "kik_sku_count": {"type": ["integer", "null"]},
            "kik_share_percent": {"type": ["integer", "null"]},
            "fill_level_percent": {"type": ["integer", "null"]},
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
            "has_monobrand_block": {"type": ["boolean", "null"]},
            "has_foreign_label": {"type": ["boolean", "null"]},
            "has_non_icecream_products": {"type": ["boolean", "null"]},
            "has_empty_sections": {"type": ["boolean", "null"]},
            "is_kik_mixed_with_competitors": {"type": ["boolean", "null"]},
            "status_score": {"type": ["integer", "null"]},
            "confidence_score": {"type": ["integer", "null"]},
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


def call_model(
    api_key: str,
    model: str,
    target_image_path: Path,
    reference_items: list[tuple[str, Path]],
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

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": JSON_SCHEMA,
        },
    }

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
            model_output = extract_json_from_response(parsed)
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

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in exts])
    if not image_paths:
        raise FileNotFoundError(f"No images found in {INPUT_DIR}.")

    reference_items = _reference_items()
    if not reference_items:
        print(f"[WARN] No reference images found in {REFERENCE_DIR.as_posix()}. Running without references.")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    total = len(MODELS) * len(image_paths)
    with tqdm(total=total, desc="Running VLM eval") as pbar:
        for model in MODELS:
            for image_path in image_paths:
                output, error, latency_sec = call_model(api_key, model, image_path, reference_items)

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
