from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ...data import EvalCase, find_images
from .schema import KIK_REQUIRED_FIELDS

DEFAULT_MANUAL_GT = Path("data/ground_truth/manual_ground_truth.jsonl")
DEFAULT_CSV_GT = Path("data/ground_truth/kik_report_ground_truth.csv")


def resolve_kik_labels_path(cli_labels: Path | None, project_root: Path | None = None) -> tuple[Path, str]:
    root = project_root or Path(".")
    if cli_labels is not None:
        labels = cli_labels if cli_labels.is_absolute() else root / cli_labels
        if not labels.exists():
            raise FileNotFoundError(f"KIK labels file not found: {labels}")
        return labels, "explicit"

    manual = root / DEFAULT_MANUAL_GT
    if manual.exists():
        return manual, "manual_ground_truth_jsonl"

    csv_path = root / DEFAULT_CSV_GT
    if csv_path.exists():
        generate_kik_jsonl_from_csv(csv_path, manual)
        return manual, "generated_from_kik_report_ground_truth_csv"

    raise FileNotFoundError(
        "KIK labels not found. Expected data/ground_truth/manual_ground_truth.jsonl or "
        "data/ground_truth/kik_report_ground_truth.csv"
    )


def load_kik_labels(path: Path) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = row.get("image_id") or row.get("image")
            if not isinstance(image_id, str) or not image_id.strip():
                raise ValueError(f"{path}:{line_no}: image_id must be string")
            labels[image_id] = normalize_kik_expected(row)
    return labels


def load_kik_cases(images_dir: Path, labels_path: Path, limit: int | None = None) -> list[EvalCase]:
    labels = load_kik_labels(labels_path)
    images = find_images(images_dir)
    cases: list[EvalCase] = []
    for image, expected in labels.items():
        image_path = images.get(image)
        if image_path is None:
            continue
        cases.append(EvalCase(image=image, image_path=image_path, expected=expected))
        if limit and len(cases) >= limit:
            break
    return cases


def generate_kik_jsonl_from_csv(csv_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as input_handle, output_path.open(
        "w", encoding="utf-8"
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        if "image_id" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} missing required image_id column")
        for row in reader:
            obj = row_to_kik_jsonl_row(row)
            output_handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return output_path


def row_to_kik_jsonl_row(row: dict[str, Any]) -> dict[str, Any]:
    image_id = _norm_str(row.get("image_id"))
    if not image_id:
        raise ValueError("Missing image_id in KIK ground truth CSV row")
    out = {"image_id": image_id}
    out.update(normalize_kik_expected(row))
    return out


def normalize_kik_expected(row: dict[str, Any]) -> dict[str, Any]:
    expected: dict[str, Any] = {}
    crop_full = _crop_is_full(row.get("photo_crop_quality"))

    source = dict(row)
    source.setdefault("photo_crop_is_full", crop_full)
    if "has_monobrand_block" not in source:
        source["has_monobrand_block"] = source.get("has_kik_grouped_block")

    for field in KIK_REQUIRED_FIELDS:
        if field in {"status_score", "kik_sku_count", "kik_share_percent"}:
            expected[field] = _parse_int_field(field, source.get(field), source)
        elif field == "has_poleno_or_briquette":
            expected[field] = _parse_poleno_or_briquette(source)
        else:
            expected[field] = _parse_bool(source.get(field))
    return expected


def _parse_int_field(field: str, value: Any, source: dict[str, Any]) -> int | None:
    if field == "status_score" and _is_blank(value):
        return {"normal": 0, "attention": 1, "critical": 2}.get(_norm_str(source.get("status")).lower())
    return _parse_int(value)


def _crop_is_full(value: Any) -> bool | None:
    text = _norm_str(value).lower()
    if text in {"", "unknown", "null", "none", "nan", "-"}:
        return None
    if text == "full":
        return True
    if text in {"partial", "bad"}:
        return False
    return None


def _parse_poleno_or_briquette(source: dict[str, Any]) -> bool | None:
    values = [
        _parse_bool(source.get("has_poleno_or_briquette")),
        _parse_bool(source.get("has_poleno")),
        _parse_bool(source.get("has_briquette")),
    ]
    if any(value is True for value in values):
        return True
    if any(value is False for value in values):
        return False
    return None


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _norm_str(value).lower()
    if text in {"", "unknown", "null", "none", "nan", "-"}:
        return None
    if text in {"true", "1", "yes", "y", "да", "истина"}:
        return True
    if text in {"false", "0", "no", "n", "нет", "ложь"}:
        return False
    return None


def _parse_int(value: Any) -> int | None:
    if isinstance(value, bool) or _is_blank(value):
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _is_blank(value: Any) -> bool:
    return _norm_str(value).lower() in {"", "unknown", "null", "none", "nan", "-"}


def _norm_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
