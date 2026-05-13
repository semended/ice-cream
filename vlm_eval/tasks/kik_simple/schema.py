from __future__ import annotations

from dataclasses import dataclass
from typing import Any

KIK_SIMPLE_REQUIRED_FIELDS = [
    "is_trade_equipment_photo",
    "is_ice_cream_equipment",
    "kik_present",
    "kik_sku_count",
    "kik_share_percent",
    "has_monobrand_block",
    "has_non_icecream_products",
    "is_kik_mixed_with_competitors",
    "status_score",
]

KIK_SIMPLE_BOOLEAN_FIELDS = [
    "is_trade_equipment_photo",
    "is_ice_cream_equipment",
    "kik_present",
    "has_monobrand_block",
    "has_non_icecream_products",
    "is_kik_mixed_with_competitors",
]

KIK_SIMPLE_NULLABLE_BOOLEAN_FIELDS = {
    "has_monobrand_block",
    "has_non_icecream_products",
    "is_kik_mixed_with_competitors",
}

KIK_SIMPLE_NUMERIC_FIELDS = [
    "kik_sku_count",
    "kik_share_percent",
    "status_score",
]

KIK_SIMPLE_INTEGER_RANGES = {
    "status_score": (0, 2),
    "kik_sku_count": (0, 100),
    "kik_share_percent": (0, 100),
}

KIK_SIMPLE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": KIK_SIMPLE_REQUIRED_FIELDS,
    "properties": {
        "is_trade_equipment_photo": {"type": "boolean"},
        "is_ice_cream_equipment": {"type": "boolean"},
        "kik_present": {"type": "boolean"},
        "kik_sku_count": {"type": "integer", "minimum": 0, "maximum": 100},
        "kik_share_percent": {"type": "integer", "minimum": 0, "maximum": 100},
        "has_monobrand_block": {"type": ["boolean", "null"]},
        "has_non_icecream_products": {"type": ["boolean", "null"]},
        "is_kik_mixed_with_competitors": {"type": ["boolean", "null"]},
        "status_score": {"type": "integer", "enum": [0, 1, 2]},
    },
}


@dataclass
class KikSimpleValidationResult:
    ok: bool
    errors: list[str]


def response_format_json_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "kik_simple_sku_lookup_eval",
            "strict": True,
            "schema": KIK_SIMPLE_JSON_SCHEMA,
        },
    }


def validate_kik_simple_prediction(obj: dict[str, Any]) -> KikSimpleValidationResult:
    errors: list[str] = []
    missing = [field for field in KIK_SIMPLE_REQUIRED_FIELDS if field not in obj]
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")

    extra = [field for field in obj if field not in KIK_SIMPLE_REQUIRED_FIELDS]
    if extra:
        errors.append(f"unexpected fields: {', '.join(extra)}")

    for field in KIK_SIMPLE_BOOLEAN_FIELDS:
        if field not in obj:
            continue
        value = obj[field]
        if field in KIK_SIMPLE_NULLABLE_BOOLEAN_FIELDS and value is None:
            continue
        if not isinstance(value, bool):
            errors.append(f"{field} must be boolean" + (" or null" if field in KIK_SIMPLE_NULLABLE_BOOLEAN_FIELDS else ""))

    for field, (minimum, maximum) in KIK_SIMPLE_INTEGER_RANGES.items():
        value = obj.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
            errors.append(f"{field} must be integer {minimum}..{maximum}")

    return KikSimpleValidationResult(ok=not errors, errors=errors)


def make_mock_kik_simple_prediction() -> dict[str, Any]:
    return {
        "is_trade_equipment_photo": True,
        "is_ice_cream_equipment": True,
        "kik_present": True,
        "kik_sku_count": 1,
        "kik_share_percent": 10,
        "has_monobrand_block": False,
        "has_non_icecream_products": False,
        "is_kik_mixed_with_competitors": False,
        "status_score": 1,
    }
