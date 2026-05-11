from __future__ import annotations

from dataclasses import dataclass
from typing import Any

KIK_REQUIRED_FIELDS = [
    "is_trade_equipment_photo",
    "is_ice_cream_equipment",
    "photo_crop_is_full",
    "kik_present",
    "kik_sku_count",
    "kik_share_percent",
    "has_cup",
    "has_eskimo",
    "has_lakomka",
    "has_cone",
    "has_sandwich",
    "has_bucket",
    "has_poleno_or_briquette",
    "has_posm",
    "has_monobrand_block",
    "has_foreign_label",
    "has_non_icecream_products",
    "has_empty_sections",
    "is_kik_mixed_with_competitors",
    "status_score",
]

KIK_BOOLEAN_FIELDS = [
    "is_trade_equipment_photo",
    "is_ice_cream_equipment",
    "photo_crop_is_full",
    "kik_present",
    "has_cup",
    "has_eskimo",
    "has_lakomka",
    "has_cone",
    "has_sandwich",
    "has_bucket",
    "has_poleno_or_briquette",
    "has_posm",
    "has_monobrand_block",
    "has_foreign_label",
    "has_non_icecream_products",
    "has_empty_sections",
    "is_kik_mixed_with_competitors",
]

KIK_NULLABLE_BOOLEAN_FIELDS = {
    "has_cup",
    "has_eskimo",
    "has_lakomka",
    "has_cone",
    "has_sandwich",
    "has_bucket",
    "has_poleno_or_briquette",
    "has_posm",
    "has_monobrand_block",
    "has_foreign_label",
    "has_non_icecream_products",
    "has_empty_sections",
    "is_kik_mixed_with_competitors",
}

KIK_NUMERIC_FIELDS = [
    "kik_sku_count",
    "kik_share_percent",
    "status_score",
]

KIK_INTEGER_RANGES = {
    "status_score": (0, 2),
    "kik_sku_count": (0, 30),
    "kik_share_percent": (0, 100),
}

KIK_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": KIK_REQUIRED_FIELDS,
    "properties": {
        "is_trade_equipment_photo": {"type": "boolean"},
        "is_ice_cream_equipment": {"type": "boolean"},
        "photo_crop_is_full": {"type": "boolean"},
        "kik_present": {"type": "boolean"},
        "kik_sku_count": {"type": "integer", "minimum": 0, "maximum": 30},
        "kik_share_percent": {"type": "integer", "minimum": 0, "maximum": 100},
        "has_cup": {"type": ["boolean", "null"]},
        "has_eskimo": {"type": ["boolean", "null"]},
        "has_lakomka": {"type": ["boolean", "null"]},
        "has_cone": {"type": ["boolean", "null"]},
        "has_sandwich": {"type": ["boolean", "null"]},
        "has_bucket": {"type": ["boolean", "null"]},
        "has_poleno_or_briquette": {"type": ["boolean", "null"]},
        "has_posm": {"type": ["boolean", "null"]},
        "has_monobrand_block": {"type": ["boolean", "null"]},
        "has_foreign_label": {"type": ["boolean", "null"]},
        "has_non_icecream_products": {"type": ["boolean", "null"]},
        "has_empty_sections": {"type": ["boolean", "null"]},
        "is_kik_mixed_with_competitors": {"type": ["boolean", "null"]},
        "status_score": {"type": "integer", "enum": [0, 1, 2]},
    },
}


@dataclass
class KikValidationResult:
    ok: bool
    errors: list[str]


def response_format_json_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "kik_retail_execution_eval",
            "strict": True,
            "schema": KIK_JSON_SCHEMA,
        },
    }


def validate_kik_prediction(obj: dict[str, Any]) -> KikValidationResult:
    errors: list[str] = []
    missing = [field for field in KIK_REQUIRED_FIELDS if field not in obj]
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")

    extra = [field for field in obj if field not in KIK_REQUIRED_FIELDS]
    if extra:
        errors.append(f"unexpected fields: {', '.join(extra)}")

    for field in KIK_BOOLEAN_FIELDS:
        if field not in obj:
            continue
        value = obj[field]
        if field in KIK_NULLABLE_BOOLEAN_FIELDS and value is None:
            continue
        if not isinstance(value, bool):
            errors.append(f"{field} must be boolean" + (" or null" if field in KIK_NULLABLE_BOOLEAN_FIELDS else ""))

    for field, (minimum, maximum) in KIK_INTEGER_RANGES.items():
        value = obj.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
            errors.append(f"{field} must be integer {minimum}..{maximum}")

    return KikValidationResult(ok=not errors, errors=errors)


def make_mock_kik_prediction() -> dict[str, Any]:
    return {
        "is_trade_equipment_photo": True,
        "is_ice_cream_equipment": True,
        "photo_crop_is_full": True,
        "kik_present": True,
        "kik_sku_count": 1,
        "kik_share_percent": 10,
        "has_cup": True,
        "has_eskimo": False,
        "has_lakomka": False,
        "has_cone": False,
        "has_sandwich": False,
        "has_bucket": False,
        "has_poleno_or_briquette": False,
        "has_posm": False,
        "has_monobrand_block": False,
        "has_foreign_label": None,
        "has_non_icecream_products": False,
        "has_empty_sections": False,
        "is_kik_mixed_with_competitors": False,
        "status_score": 1,
    }
