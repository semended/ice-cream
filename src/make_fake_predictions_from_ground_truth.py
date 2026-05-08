import csv
import json
from pathlib import Path
from typing import Any


GROUND_TRUTH_JSONL = Path("data/ground_truth/manual_ground_truth.jsonl")
OUTPUT_CSV = Path("results/openrouter_eval_results.csv")

MODEL_NAME = "fake_perfect_model"

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


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def main() -> None:
    if not GROUND_TRUTH_JSONL.exists():
        raise FileNotFoundError(f"Ground truth JSONL not found: {GROUND_TRUTH_JSONL.as_posix()}")

    rows: list[dict[str, Any]] = []
    with GROUND_TRUTH_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ground_truth = json.loads(line)
            image_id = ground_truth.get("image_id")
            if not image_id:
                raise ValueError("Ground truth row is missing image_id")

            prediction = {field: ground_truth.get(field) for field in PREDICTION_FIELDS}
            if prediction.get("has_kik_grouped_block") is None:
                prediction["has_kik_grouped_block"] = ground_truth.get("has_monobrand_block")
            row = {
                "model": MODEL_NAME,
                "image_id": image_id,
                "latency_sec": 0,
                "error": "",
                "prediction_json": json.dumps(prediction, ensure_ascii=False),
            }
            for field in PREDICTION_FIELDS:
                row[field] = csv_value(prediction.get(field))
            rows.append(row)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "image_id", "latency_sec", "error", "prediction_json"] + PREDICTION_FIELDS
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Input JSONL: {GROUND_TRUTH_JSONL.as_posix()}")
    print(f"Processed rows: {len(rows)}")
    print(f"Saved fake predictions: {OUTPUT_CSV.as_posix()}")
    print(f"Model: {MODEL_NAME}")


if __name__ == "__main__":
    main()
