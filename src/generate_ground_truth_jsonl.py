import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GROUND_TRUTH_CSV = PROJECT_ROOT / "data" / "ground_truth" / "kik_report_ground_truth.csv"
OUTPUT_JSONL = PROJECT_ROOT / "data" / "ground_truth" / "manual_ground_truth.jsonl"


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
    "has_monobrand_block",
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
    "status_score",
    "confidence_score",
]

OUTPUT_FIELDS = ["image_id"] + BOOLEAN_FIELDS + NUMERIC_FIELDS + ["uncertainty_notes"]


def _norm_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()


def parse_bool(x: Any) -> bool | None:
    s = _norm_str(x).lower()
    if s in ("", "unknown", "null", "none", "nan", "-"):
        return None
    if s in ("true", "1", "yes", "y", "да", "истина"):
        return True
    if s in ("false", "0", "no", "n", "нет", "ложь"):
        return False
    return None


def parse_int(x: Any) -> int | None:
    s = _norm_str(x)
    if s == "":
        return None
    if s.lower() in ("unknown", "null", "none", "nan", "-"):
        return None
    try:
        # handles "10.0" from CSV as well
        val = int(float(s))
    except Exception:  # noqa: BLE001
        return None
    return val


def map_photo_quality_score(x: Any) -> int | None:
    s = _norm_str(x).lower()
    return {"good": 2, "medium": 1, "bad": 0}.get(s)


def map_analysis_possible_score(x: Any) -> int | None:
    s = _norm_str(x).lower()
    return {"true": 2, "partial": 1, "false": 0}.get(s)


def map_status_score(x: Any) -> int | None:
    s = _norm_str(x).lower()
    return {"normal": 0, "attention": 1, "critical": 2}.get(s)


def map_confidence_score(x: Any) -> int | None:
    s = _norm_str(x).lower()
    return {"high": 2, "medium": 1, "low": 0}.get(s)


def map_equipment_flags(equipment_type: Any) -> dict[str, bool | None]:
    s = _norm_str(equipment_type).lower()
    if s in ("", "unknown", "null", "none", "nan", "-"):
        return {
            "equipment_is_open_freezer": None,
            "equipment_is_vertical_fridge": None,
            "equipment_is_display_freezer": None,
        }

    is_open = "open_freezer" in s or ("open" in s and "freezer" in s)
    is_vertical = "vertical_fridge" in s or ("vertical" in s and ("fridge" in s or "холодильник" in s))
    is_display = "display_freezer" in s or ("display" in s and "freezer" in s)

    # If legacy values are in Russian, keep simple heuristic.
    if not any([is_open, is_vertical, is_display]):
        is_open = "лар" in s
        is_vertical = "вертик" in s
        is_display = "витрин" in s

    return {
        "equipment_is_open_freezer": bool(is_open),
        "equipment_is_vertical_fridge": bool(is_vertical),
        "equipment_is_display_freezer": bool(is_display),
    }


def map_photo_crop_flags(photo_crop_quality: Any) -> tuple[bool | None, bool | None]:
    s = _norm_str(photo_crop_quality).lower()
    if s in ("", "unknown", "null", "none", "nan", "-"):
        return None, None
    if s == "full":
        return True, False
    if s == "partial":
        return False, True
    if s == "bad":
        return False, False
    return None, None


def to_uncertainty_notes(row: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    for key in ("violations", "notes", "visible_sku"):
        s = _norm_str(row.get(key))
        if s and s.lower() not in ("unknown", "null", "none", "nan", "-"):
            notes.append(f"{key}: {s}")
    return notes


def row_to_countable_dict(row: dict[str, Any]) -> dict[str, Any]:
    image_id = _norm_str(row.get("image_id"))
    if not image_id:
        raise ValueError("Missing image_id in ground truth row")

    out: dict[str, Any] = {"image_id": image_id}
    out["is_trade_equipment_photo"] = parse_bool(row.get("is_trade_equipment_photo"))
    out["is_ice_cream_equipment"] = parse_bool(row.get("is_ice_cream_equipment"))

    out.update(map_equipment_flags(row.get("equipment_type")))
    out["equipment_is_branded"] = parse_bool(row.get("branded_equipment"))

    out["photo_quality_score"] = map_photo_quality_score(row.get("photo_quality"))
    crop_full, crop_partial = map_photo_crop_flags(row.get("photo_crop_quality"))
    out["photo_crop_is_full"] = crop_full
    out["photo_crop_is_partial"] = crop_partial
    out["analysis_possible_score"] = map_analysis_possible_score(row.get("analysis_possible"))

    out["kik_present"] = parse_bool(row.get("kik_present"))
    out["kik_sku_count"] = parse_int(row.get("kik_sku_count"))
    out["kik_share_percent"] = parse_int(row.get("kik_share_percent"))
    out["fill_level_percent"] = parse_int(row.get("fill_level_percent"))

    for f in [
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
    ]:
        out[f] = parse_bool(row.get(f))

    out["status_score"] = map_status_score(row.get("status"))
    out["confidence_score"] = map_confidence_score(row.get("confidence"))
    out["uncertainty_notes"] = to_uncertainty_notes(row)

    # strict shape
    for f in OUTPUT_FIELDS:
        out.setdefault(f, None if f != "uncertainty_notes" else [])
    return out


def main() -> None:
    if not GROUND_TRUTH_CSV.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {GROUND_TRUTH_CSV}")

    df = pd.read_csv(GROUND_TRUTH_CSV, dtype=str, keep_default_na=False)
    required_min = ["image_id"]
    missing = [c for c in required_min if c not in df.columns]
    if missing:
        raise ValueError("manual_ground_truth.csv missing required columns: " + ", ".join(missing))

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            obj = row_to_countable_dict(r.to_dict())
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1

    print(f"Saved: {OUTPUT_JSONL.as_posix()} ({written} rows)")


if __name__ == "__main__":
    main()

