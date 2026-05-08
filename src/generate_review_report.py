import ast
import csv
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(".")
GT_JSONL = ROOT / "data/ground_truth/manual_ground_truth.jsonl"
DETAILS_CSV = ROOT / "results/model_comparison_details.csv"
WORST_CSV = ROOT / "results/worst_cases_by_model.csv"
SUMMARY_CSV = ROOT / "results/model_comparison_summary.csv"
OUTPUT_CSV = ROOT / "results/model_answers_compact.csv"
OUTPUT_LONG_CSV = ROOT / "results/business_comparison_long.csv"
OUTPUT_WIDE_CSV = ROOT / "results/business_comparison_wide.csv"
OUTPUT_HTML = ROOT / "results/review_report.html"
REAL_IMAGES_DIR = ROOT / "data/real_images"
REFERENCE_DIR = ROOT / "data/reference_images"
RUN_SCRIPT = ROOT / "src/run_openrouter_eval.py"

CORE_FIELDS = [
    "kik_present",
    "kik_sku_count",
    "kik_share_percent",
]
CATEGORY_FIELDS = [
    "has_cup",
    "has_cone",
    "has_eskimo",
    "has_lakomka",
    "has_sandwich",
    "has_large_pack",
    "has_bucket",
    "has_poleno",
    "has_briquette",
]
BLOCK_FIELDS = [
    "has_kik_grouped_block",
    "has_kik_products_outside_block",
    "kik_outside_block_severity",
    "is_kik_mixed_with_competitors",
]
VISIBLE_FIELDS = CORE_FIELDS + CATEGORY_FIELDS + BLOCK_FIELDS
BOOLEAN_FIELDS = {
    "kik_present",
    "has_posm",
    "has_cup",
    "has_cone",
    "has_eskimo",
    "has_lakomka",
    "has_sandwich",
    "has_large_pack",
    "has_bucket",
    "has_poleno",
    "has_briquette",
    "has_kik_grouped_block",
    "has_kik_products_outside_block",
    "is_kik_mixed_with_competitors",
}
NUMERIC_FIELDS = {
    "kik_sku_count",
    "kik_share_percent",
    "kik_outside_block_severity",
}
LABELS = {
    "kik_present": "KIK",
    "kik_sku_count": "SKU",
    "kik_share_percent": "Share",
    "fill_level_percent": "Fill",
    "has_posm": "POSM",
    "has_cup": "Cup",
    "has_cone": "Cone",
    "has_eskimo": "Eskimo",
    "has_lakomka": "Lakomka",
    "has_sandwich": "Sandwich",
    "has_large_pack": "Large pack",
    "has_bucket": "Bucket",
    "has_poleno": "Poleno",
    "has_briquette": "Briquette",
    "has_kik_grouped_block": "Grouped block",
    "has_kik_products_outside_block": "Outside block",
    "is_kik_mixed_with_competitors": "Mixed",
    "kik_outside_block_severity": "Outside severity",
    "status_score": "Status",
}
COMPACT_COLUMNS = [
    "image_id",
    "model",
    "overall_verdict",
    "gt_core",
    "pred_core",
    "critical_mistakes",
    "numeric_mistakes",
    "error_score",
    "short_comment",
]
LONG_COLUMNS = [
    "image_id",
    "image_path",
    "case_id",
    "source_type",
    "model_name",
    "model_type",
    *VISIBLE_FIELDS,
    "json_valid",
    "latency_sec",
    "error_message",
]
WIDE_BASE_COLUMNS = ["image_id", "image_path", "case_id"]
STATUS = {0: "normal", 1: "attention", 2: "critical"}
TOLERANCE = {"kik_sku_count": 2, "kik_share_percent": 10, "fill_level_percent": 10, "kik_outside_block_severity": 0}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_gt() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not GT_JSONL.exists():
        return rows
    with GT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row["image_id"]] = row
    return rows


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "да"}:
        return True
    if text in {"false", "0", "no", "n", "нет"}:
        return False
    return None


def parse_int(value: Any) -> int | None:
    parsed = parse_float(value)
    return int(parsed) if parsed is not None else None


def is_openai_model(model: str) -> bool:
    name = model.lower()
    return "openai" in name or "gpt-4o" in name or "gpt-4.1" in name


def valid_candidate(row: dict[str, str]) -> bool:
    total_cases = parse_float(row.get("total_cases"))
    errors = parse_float(row.get("api_or_json_errors")) or 0
    coverage = parse_float(row.get("coverage_score"))
    mvp = parse_float(row.get("mvp_score"))
    if total_cases is not None and errors == total_cases:
        return False
    if coverage == 0:
        return False
    if mvp is None:
        has_fallback = any(parse_float(row.get(c)) is not None for c in ["boolean_macro_accuracy", "numeric_score", "coverage_score"])
        if not has_fallback:
            return False
    return True


def sort_key(row: dict[str, str]) -> tuple[float, float, float, float]:
    final_score = parse_float(row.get("final_score_100"))
    if final_score is not None:
        return (1.0, final_score, parse_float(row.get("s_present")) or -1, parse_float(row.get("s_share")) or -1)
    mvp = parse_float(row.get("mvp_score"))
    if mvp is not None:
        return (1.0, mvp, parse_float(row.get("boolean_macro_accuracy")) or -1, parse_float(row.get("numeric_score")) or -1)
    return (
        0.0,
        parse_float(row.get("boolean_macro_accuracy")) or -1,
        parse_float(row.get("numeric_score")) or -1,
        parse_float(row.get("coverage_score")) or -1,
    )


def select_models(summary: list[dict[str, str]]) -> tuple[str | None, str | None]:
    valid_rows = [row for row in summary if valid_candidate(row)]
    openai_rows = [row for row in valid_rows if is_openai_model(row.get("model", ""))]
    self_host_rows = [row for row in valid_rows if not is_openai_model(row.get("model", ""))]
    openai = max(openai_rows, key=sort_key).get("model") if openai_rows else None
    self_host = max(self_host_rows, key=sort_key).get("model") if self_host_rows else None
    return openai, self_host


def bool_text(value: Any) -> str:
    parsed = parse_bool(value)
    if parsed is True:
        return "yes"
    if parsed is False:
        return "no"
    return "null"


def int_text(value: Any, suffix: str = "") -> str:
    parsed = parse_int(value)
    if parsed is None:
        return "null"
    return f"{parsed}{suffix}"


def status_text(value: Any) -> str:
    parsed = parse_int(value)
    if parsed is None:
        return "null"
    return STATUS.get(parsed, str(parsed))


def core_value_text(field: str, value: Any) -> str:
    if field in BOOLEAN_FIELDS:
        return bool_text(value)
    if field in {"kik_share_percent", "fill_level_percent"}:
        return int_text(value, "%")
    if field == "status_score":
        return status_text(value)
    return int_text(value)


def compact_core(row: dict[str, Any], suffix: str) -> str:
    return "; ".join(
        [
            f"KIK={core_value_text('kik_present', row.get('kik_present' + suffix))}",
            f"SKU={core_value_text('kik_sku_count', row.get('kik_sku_count' + suffix))}",
            f"share={core_value_text('kik_share_percent', row.get('kik_share_percent' + suffix))}",
            f"grouped={core_value_text('has_kik_grouped_block', row.get('has_kik_grouped_block' + suffix))}",
            f"outside={core_value_text('has_kik_products_outside_block', row.get('has_kik_products_outside_block' + suffix))}",
            f"mixed={core_value_text('is_kik_mixed_with_competitors', row.get('is_kik_mixed_with_competitors' + suffix))}",
            f"outside_sev={core_value_text('kik_outside_block_severity', row.get('kik_outside_block_severity' + suffix))}",
        ]
    )


def typed_value(field: str, value: Any) -> Any:
    if field in BOOLEAN_FIELDS:
        return parse_bool(value)
    return parse_int(value)


def field_state(field: str, gt: Any, pred: Any) -> str:
    if gt is None:
        return "unknown"
    if pred is None:
        return "missing"
    if field in BOOLEAN_FIELDS or field == "status_score":
        return "correct" if gt == pred else "wrong"
    if field in NUMERIC_FIELDS:
        diff = abs(gt - pred)
        if diff == 0:
            return "correct"
        if diff <= TOLERANCE[field]:
            return "near"
        return "wrong"
    return "correct" if gt == pred else "wrong"


def analyze_row(row: dict[str, str]) -> tuple[str, list[str], list[str], dict[str, str], str]:
    critical: list[str] = []
    numeric: list[str] = []
    states: dict[str, str] = {}
    for field in VISIBLE_FIELDS:
        gt = typed_value(field, row.get(f"{field}_gt"))
        pred = typed_value(field, row.get(f"{field}_pred"))
        state = field_state(field, gt, pred)
        states[field] = state
        if state in {"wrong", "missing"}:
            critical.append(f"{LABELS[field]}: {core_value_text(field, gt)} -> {core_value_text(field, pred)}")
        if field in NUMERIC_FIELDS and gt is not None and pred is not None and gt != pred:
            numeric.append(f"{LABELS[field]}: {core_value_text(field, gt)} -> {core_value_text(field, pred)}")
    error = str(row.get("error", "")).strip()
    if error:
        verdict = "failed"
    else:
        hard_errors = sum(1 for f in ["kik_present", "status_score"] if states.get(f) in {"wrong", "missing"})
        critical_wrong_count = sum(1 for state in states.values() if state in {"wrong", "missing"})
        if hard_errors > 0 or critical_wrong_count >= 3:
            verdict = "bad"
        elif critical_wrong_count > 0:
            verdict = "partial"
        else:
            verdict = "good"
    short_comment = "OK"
    if verdict == "failed":
        short_comment = error[:180]
    elif critical:
        short_comment = "; ".join(critical[:3])
        if len(critical) > 3:
            short_comment += f"; +{len(critical) - 3} more"
    elif numeric:
        short_comment = "; ".join(numeric[:3])
    return verdict, critical, numeric, states, short_comment


def numeric_delta_text(field: str, gt: Any, pred: Any) -> str:
    if gt is None or pred is None:
        return ""
    delta = pred - gt
    sign = "+" if delta > 0 else ""
    unit = " pp" if field in {"kik_share_percent", "fill_level_percent"} else ""
    direction = "overestimated" if delta > 0 else "underestimated"
    return f"{direction} {LABELS[field]} by {sign}{delta}{unit}"


def mismatch_summary(model_label: str, row: dict[str, str]) -> list[str]:
    missed: list[str] = []
    wrong: list[str] = []
    numeric_notes: list[str] = []
    for field in VISIBLE_FIELDS:
        gt = typed_value(field, row.get(f"{field}_gt"))
        pred = typed_value(field, row.get(f"{field}_pred"))
        state = field_state(field, gt, pred)
        if state in {"correct", "unknown"}:
            continue
        if field in NUMERIC_FIELDS and gt is not None and pred is not None:
            note = numeric_delta_text(field, gt, pred)
            if note:
                numeric_notes.append(note)
        elif pred is None:
            missed.append(LABELS[field])
        else:
            wrong.append(LABELS[field])

    out: list[str] = []
    if missed:
        out.append(f"{model_label} missed: {', '.join(missed)}")
    if wrong:
        out.append(f"{model_label} wrong: {', '.join(wrong)}")
    out.extend(f"{model_label} {note}" for note in numeric_notes)
    if row.get("error", "").strip():
        out.append(f"{model_label} API/JSON error: {row['error'][:180]}")
    return out


def gt_comment(gt_row: dict[str, Any]) -> str:
    notes = gt_row.get("uncertainty_notes") or []
    comments = []
    for note in notes:
        text = str(note)
        if text.startswith("comment: "):
            comments.append(text[len("comment: ") :])
    return " ".join(comments) or " | ".join(str(x) for x in notes)


def prompt_text() -> str:
    try:
        tree = ast.parse(RUN_SCRIPT.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "PROMPT":
                        return ast.literal_eval(node.value)
    except Exception as exc:  # noqa: BLE001
        return f"Could not extract PROMPT: {exc}"
    return "PROMPT not found"


def rel_from_results(path: Path) -> str:
    return "../" + path.as_posix()


def image_for(image_id: str) -> Path | None:
    direct = REAL_IMAGES_DIR / image_id
    if direct.exists():
        return direct
    stem = Path(image_id).stem
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = REAL_IMAGES_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def image_path_for_csv(image_id: str) -> str:
    img = image_for(image_id)
    return img.as_posix() if img else ""


def reference_images() -> list[Path]:
    if not REFERENCE_DIR.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in REFERENCE_DIR.iterdir() if p.suffix.lower() in exts)


def pretty_json(raw: str) -> str:
    if not raw:
        return ""
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
    except Exception:  # noqa: BLE001
        return raw


def source_prefix(model_type: str) -> str:
    return {"ground_truth": "gt", "openai": "openai", "self_host_candidate": "self_host"}[model_type]


def value_for_output(field: str, value: Any) -> Any:
    if field in BOOLEAN_FIELDS:
        parsed_bool = parse_bool(value)
        if parsed_bool is None:
            return ""
        return str(parsed_bool).lower()
    parsed_int = parse_int(value)
    return "" if parsed_int is None else parsed_int


def row_value(row: dict[str, Any], field: str, suffix: str) -> Any:
    return value_for_output(field, row.get(field + suffix))


def gt_value(gt_row: dict[str, Any], field: str) -> Any:
    return value_for_output(field, gt_row.get(field))


def build_long_rows(
    image_ids: list[str],
    gt: dict[str, dict[str, Any]],
    by_image: dict[str, list[dict[str, str]]],
    openai_model: str | None,
    self_host_model: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image_id in image_ids:
        image_path = image_path_for_csv(image_id)
        case_id = Path(image_id).stem
        gt_row = gt.get(image_id, {})
        base = {
            "image_id": image_id,
            "image_path": image_path,
            "case_id": case_id,
            "source_type": "Ground Truth",
            "model_name": "Ground Truth",
            "model_type": "ground_truth",
            "json_valid": "true",
            "latency_sec": "",
            "error_message": "",
        }
        for field in VISIBLE_FIELDS:
            base[field] = gt_value(gt_row, field)
        rows.append(base)

        for model_type, model_name in [("openai", openai_model), ("self_host_candidate", self_host_model)]:
            if not model_name:
                continue
            match = next((row for row in by_image.get(image_id, []) if row.get("model") == model_name), None)
            if not match:
                continue
            error = match.get("error", "").strip()
            pred = {
                "image_id": image_id,
                "image_path": image_path,
                "case_id": case_id,
                "source_type": "Model",
                "model_name": model_name,
                "model_type": model_type,
                "json_valid": "false" if error else "true",
                "latency_sec": match.get("latency_sec", ""),
                "error_message": error,
            }
            for field in VISIBLE_FIELDS:
                pred[field] = row_value(match, field, "_pred")
            rows.append(pred)
    return rows


def numeric_error_sum(row: dict[str, str]) -> float:
    total = 0.0
    for field in NUMERIC_FIELDS:
        gt = typed_value(field, row.get(f"{field}_gt"))
        pred = typed_value(field, row.get(f"{field}_pred"))
        if gt is not None and pred is not None:
            total += abs(gt - pred)
    return total


def boolean_mismatch_count(row: dict[str, str]) -> int:
    count = 0
    for field in BOOLEAN_FIELDS.intersection(VISIBLE_FIELDS):
        gt = typed_value(field, row.get(f"{field}_gt"))
        pred = typed_value(field, row.get(f"{field}_pred"))
        if gt is not None and pred != gt:
            count += 1
    return count


def build_wide_rows(
    image_ids: list[str],
    gt: dict[str, dict[str, Any]],
    by_image: dict[str, list[dict[str, str]]],
    openai_model: str | None,
    self_host_model: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image_id in image_ids:
        row: dict[str, Any] = {
            "image_id": image_id,
            "image_path": image_path_for_csv(image_id),
            "case_id": Path(image_id).stem,
            "openai_model_name": openai_model or "",
            "self_host_model_name": self_host_model or "",
        }
        gt_row = gt.get(image_id, {})
        for field in VISIBLE_FIELDS:
            row[f"gt_{field}"] = gt_value(gt_row, field)

        for model_type, model_name in [("openai", openai_model), ("self_host_candidate", self_host_model)]:
            prefix = source_prefix(model_type)
            match = next((item for item in by_image.get(image_id, []) if item.get("model") == model_name), None) if model_name else None
            for field in VISIBLE_FIELDS:
                row[f"{prefix}_{field}"] = row_value(match, field, "_pred") if match else ""
                if field in NUMERIC_FIELDS:
                    gt_num = typed_value(field, row.get(f"gt_{field}"))
                    pred_num = typed_value(field, row[f"{prefix}_{field}"])
                    row[f"{prefix}_{field}_delta"] = "" if gt_num is None or pred_num is None else pred_num - gt_num
            row[f"{prefix}_boolean_mismatch_count"] = boolean_mismatch_count(match) if match else ""
            row[f"{prefix}_numeric_error_sum"] = round(numeric_error_sum(match), 4) if match else ""
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    gt = read_gt()
    details = read_csv(DETAILS_CSV)
    worst = read_csv(WORST_CSV)
    summary = read_csv(SUMMARY_CSV)
    openai_model, self_host_model = select_models(summary)
    selected_models = {m for m in [openai_model, self_host_model] if m}
    filtered_details = [row for row in details if row.get("model") in selected_models]
    filtered_summary = [row for row in summary if row.get("model") in selected_models]
    worst_by_key = {(row.get("model"), row.get("image_id")): row for row in worst}

    compact_rows: list[dict[str, str]] = []
    analysis: dict[tuple[str, str], dict[str, Any]] = {}
    for row in filtered_details:
        model = row.get("model", "")
        image_id = row.get("image_id", "")
        worst_row = worst_by_key.get((model, image_id), {})
        verdict, critical, numeric, states, short_comment = analyze_row(row)
        compact_rows.append(
            {
                "image_id": image_id,
                "model": model,
                "overall_verdict": verdict,
                "gt_core": compact_core(row, "_gt"),
                "pred_core": compact_core(row, "_pred"),
                "critical_mistakes": "; ".join(critical),
                "numeric_mistakes": "; ".join(numeric),
                "error_score": worst_row.get("error_score", ""),
                "short_comment": short_comment,
            }
        )
        analysis[(model, image_id)] = {"verdict": verdict, "critical": critical, "numeric": numeric, "states": states}

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMPACT_COLUMNS)
        writer.writeheader()
        writer.writerows(compact_rows)

    by_image: dict[str, list[dict[str, str]]] = {}
    for row in filtered_details:
        by_image.setdefault(row.get("image_id", ""), []).append(row)
    image_ids = sorted(set(gt) | set(by_image))
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    long_rows = build_long_rows(image_ids, gt, by_image, openai_model, self_host_model)
    wide_rows = build_wide_rows(image_ids, gt, by_image, openai_model, self_host_model)
    write_csv(OUTPUT_LONG_CSV, long_rows, LONG_COLUMNS)
    write_csv(OUTPUT_WIDE_CSV, wide_rows)

    summary_cols = [
        c
        for c in [
            "model",
            "total_cases",
            "final_score_100",
            "s_present",
            "s_share",
            "s_sku",
            "s_cat",
            "s_layout",
            "CER_FN",
            "CER_FP",
            "CER_share20",
            "CER_sku",
            "presence_tp",
            "presence_fp",
            "presence_tn",
            "presence_fn",
            "kik_share_mae_pp",
            "kik_share_hit_10pp",
            "kik_sku_mae",
            "kik_sku_within_1_rate",
            "category_macro_f1",
            "json_valid_rate",
            "avg_latency_sec",
            "p95_latency_sec",
        ]
        if filtered_summary and c in filtered_summary[0]
    ]

    css = """
:root { --ink:#172033; --muted:#667085; --line:#d9dee8; --panel:#f7f8fb; --green:#dff7e8; --red:#ffe2e0; --yellow:#fff4cc; --gray:#eceff4; --blue:#eef4ff; }
* { box-sizing: border-box; }
body { margin:0; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif; color:var(--ink); background:white; }
header { padding:24px 32px; background:#101828; color:white; }
h1 { margin:0 0 8px; font-size:30px; }
h2 { margin:30px 0 14px; font-size:23px; }
h3 { margin:0 0 12px; font-size:18px; }
main { padding:24px 32px 70px; }
.muted { color:var(--muted); }
header .muted { color:#cbd5e1; }
.selection { margin-top:14px; display:grid; gap:6px; font-size:14px; }
.header-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:10px; margin-top:16px; }
.header-item { background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.18); border-radius:10px; padding:10px 12px; }
.header-item span { display:block; color:#cbd5e1; font-size:12px; margin-bottom:4px; }
.table-wrap { overflow-x:auto; border:1px solid var(--line); border-radius:10px; }
table { width:100%; border-collapse:collapse; background:white; }
th, td { padding:9px 11px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; font-size:13px; }
th { background:var(--panel); font-weight:700; white-space:nowrap; }
tr:last-child td { border-bottom:0; }
.note { background:var(--blue); border:1px solid #bfd3ff; border-radius:10px; padding:14px 16px; line-height:1.45; }
.refs { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:12px; }
.ref { border:1px solid var(--line); border-radius:10px; padding:8px; background:white; }
.ref img { width:100%; height:112px; object-fit:contain; background:var(--panel); border-radius:6px; }
.photo-section { border:1px solid var(--line); border-radius:14px; overflow:hidden; margin:28px 0; }
.photo-header { padding:14px 18px; background:#f1f4f9; display:flex; justify-content:space-between; gap:12px; align-items:center; }
.photo-header h2 { margin:0; }
.photo-grid { display:grid; grid-template-columns:minmax(280px,420px) 1fr; gap:18px; padding:18px; }
.target { width:100%; max-height:520px; object-fit:contain; border:1px solid var(--line); border-radius:10px; background:#f8fafc; }
.card { border:1px solid var(--line); border-radius:12px; background:white; padding:14px; margin-bottom:14px; }
.core-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px 12px; }
.core-item { display:flex; justify-content:space-between; gap:12px; border-bottom:1px solid #eef0f4; padding:5px 0; font-size:14px; }
.core-item span:first-child { color:var(--muted); }
.comment { margin-top:12px; padding-top:10px; border-top:1px solid var(--line); font-size:13px; line-height:1.45; }
.model-card { border:1px solid var(--line); border-radius:12px; overflow:hidden; margin:14px 0; }
.model-head { display:flex; justify-content:space-between; gap:12px; align-items:center; padding:12px 14px; background:#f8fafc; }
.model-name { font-weight:700; }
.badge { border-radius:999px; padding:3px 9px; font-size:12px; font-weight:700; text-transform:uppercase; }
.badge.good { background:var(--green); color:#087443; }
.badge.partial { background:var(--yellow); color:#8a6100; }
.badge.bad, .badge.failed { background:var(--red); color:#9b1c16; }
.state-correct { background:var(--green); }
.state-wrong { background:var(--red); font-weight:700; }
.state-missing, .state-unknown { background:var(--gray); color:#475467; }
.state-near { background:var(--yellow); }
.mistakes { padding:10px 14px; font-size:13px; line-height:1.45; color:#7a271a; }
.summary-list { margin:0; padding:10px 14px 12px 28px; font-size:13px; line-height:1.45; background:#fff7ed; color:#7c2d12; }
.field-group { margin:14px 0 6px; padding:6px 0; color:#344054; font-weight:700; font-size:14px; border-bottom:1px solid var(--line); }
details { padding:0 14px 12px; }
summary { cursor:pointer; color:var(--muted); font-size:13px; }
pre { white-space:pre-wrap; word-break:break-word; background:#111827; color:#dbeafe; padding:14px; border-radius:9px; font-size:12px; line-height:1.45; }
@media(max-width:900px){ main{padding:18px;} .photo-grid{grid-template-columns:1fr;} .core-grid{grid-template-columns:1fr;} }
"""

    parts: list[str] = []
    parts.append('<!doctype html><html><head><meta charset="utf-8"><title>VLM MVP Loom Review</title>')
    parts.append(f"<style>{css}</style></head><body>")
    parts.append("<header><h1>KIK Freezer VLM Business Review</h1>")
    parts.append('<div class="muted">Ground Truth is the human annotation. The report compares it with the selected OpenAI baseline and top-1 self-host/open-source candidate.</div>')
    parts.append('<div class="header-grid">')
    parts.append(f'<div class="header-item"><span>Generated</span><strong>{html.escape(generated_at)}</strong></div>')
    parts.append(f'<div class="header-item"><span>OpenAI baseline</span><strong>{html.escape(openai_model or "No valid OpenAI baseline available in this run")}</strong></div>')
    parts.append(f'<div class="header-item"><span>Self-host candidate</span><strong>{html.escape(self_host_model or "No valid self-host candidate available in this run")}</strong></div>')
    parts.append(f'<div class="header-item"><span>Images / cases</span><strong>{len(image_ids)}</strong></div>')
    parts.append("</div></header><main>")

    parts.append("<h2>Summary by selected model</h2>")
    if filtered_summary and summary_cols:
        parts.append('<div class="table-wrap"><table><thead><tr>')
        for col in summary_cols:
            parts.append(f"<th>{html.escape(col)}</th>")
        parts.append("</tr></thead><tbody>")
        for row in filtered_summary:
            parts.append("<tr>")
            for col in summary_cols:
                parts.append(f"<td>{html.escape(str(row.get(col, '')))}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table></div>")
    else:
        parts.append('<p class="muted">No selected model summary available.</p>')

    parts.append("<h2>What was sent to VLM</h2>")
    parts.append(
        '<div class="note">Each VLM request contained: the current audit prompt, reference images as visual examples of KIK/Renna products, one target image from <code>data/real_images</code>, and an instruction to return strict JSON.</div>'
    )

    parts.append("<h2>Current prompt</h2>")
    parts.append("<details open><summary>Prompt text</summary>")
    parts.append(f"<pre>{html.escape(prompt_text())}</pre></details>")

    parts.append("<h2>Reference images</h2>")
    refs = reference_images()
    if refs:
        parts.append('<div class="refs">')
        for p in refs:
            parts.append(f'<div class="ref"><img src="{html.escape(rel_from_results(p))}" alt="{html.escape(p.name)}"><div>{html.escape(p.name)}</div></div>')
        parts.append("</div>")
    else:
        parts.append('<p class="muted">No reference images found.</p>')

    parts.append("<h2>Photo-by-photo review</h2>")
    for image_id in image_ids:
        gt_row = gt.get(image_id, {})
        img = image_for(image_id)
        rows_for_image = by_image.get(image_id, [])
        parts.append('<section class="photo-section">')
        parts.append(f'<div class="photo-header"><h2>{html.escape(image_id)}</h2><div class="muted">{len(rows_for_image)} selected model answer(s)</div></div>')
        parts.append('<div class="photo-grid"><div>')
        if img:
            parts.append(f'<img class="target" src="{html.escape(rel_from_results(img))}" alt="{html.escape(image_id)}">')
        else:
            parts.append('<div class="muted">Target image not found</div>')
        parts.append('</div><div><div class="card"><h3>Ground truth</h3>')
        for title, fields in [("Core fields", CORE_FIELDS), ("Category indicators", CATEGORY_FIELDS), ("Block / mixed placement", BLOCK_FIELDS)]:
            parts.append(f'<div class="field-group">{html.escape(title)}</div><div class="core-grid">')
            for field in fields:
                parts.append(f'<div class="core-item"><span>{html.escape(LABELS[field])}</span><strong>{html.escape(core_value_text(field, gt_row.get(field)))}</strong></div>')
            parts.append("</div>")
        comment = gt_comment(gt_row)
        if comment:
            parts.append(f'<div class="comment"><strong>Comment:</strong> {html.escape(comment)}</div>')
        parts.append("</div></div></div>")

        parts.append('<div style="padding:0 18px 18px;">')
        for row in sorted(rows_for_image, key=lambda r: r.get("model", "")):
            key = (row.get("model", ""), image_id)
            info = analysis.get(key, {})
            verdict = str(info.get("verdict", "partial"))
            worst_row = worst_by_key.get(key, {})
            model_label = "OpenAI" if row.get("model") == openai_model else "Self-host"
            summary_items = mismatch_summary(model_label, row)
            parts.append('<div class="model-card">')
            parts.append(
                f'<div class="model-head"><div class="model-name">{html.escape(row.get("model", ""))}</div><div><span class="badge {html.escape(verdict)}">{html.escape(verdict)}</span> <span class="muted">error_score={html.escape(str(worst_row.get("error_score", "")))}</span></div></div>'
            )
            if summary_items:
                parts.append("<ul class=\"summary-list\">")
                for item in summary_items[:6]:
                    parts.append(f"<li>{html.escape(item)}</li>")
                parts.append("</ul>")
            else:
                parts.append('<div class="note" style="margin:12px 14px;">No business-field mismatches in visible fields.</div>')
            parts.append('<table><thead><tr><th>Group</th><th>Field</th><th>Ground truth</th><th>Prediction</th><th>Delta</th></tr></thead><tbody>')
            for title, fields in [("Core", CORE_FIELDS), ("Categories", CATEGORY_FIELDS), ("Block / mixed", BLOCK_FIELDS)]:
                for field in fields:
                    gt_value = typed_value(field, row.get(f"{field}_gt"))
                    pred_value = typed_value(field, row.get(f"{field}_pred"))
                    state = field_state(field, gt_value, pred_value)
                    delta = ""
                    if field in NUMERIC_FIELDS and gt_value is not None and pred_value is not None:
                        diff = pred_value - gt_value
                        delta = f"{diff:+d}" + (" pp" if field in {"kik_share_percent", "fill_level_percent"} else "")
                    parts.append("<tr>")
                    parts.append(f"<td>{html.escape(title)}</td>")
                    parts.append(f"<td>{html.escape(LABELS[field])}</td>")
                    parts.append(f"<td>{html.escape(core_value_text(field, gt_value))}</td>")
                    parts.append(f'<td class="state-{state}">{html.escape(core_value_text(field, pred_value))}</td>')
                    parts.append(f"<td>{html.escape(delta)}</td>")
                    parts.append("</tr>")
            parts.append("</tbody></table>")
            if row.get("error"):
                parts.append(f'<div class="mistakes"><strong>API/JSON error:</strong> {html.escape(row["error"])}</div>')
            raw_json = row.get("prediction_json", "")
            if raw_json:
                parts.append("<details><summary>Technical debug: raw prediction JSON</summary>")
                parts.append(f"<pre>{html.escape(pretty_json(raw_json))}</pre></details>")
            parts.append("</div>")
        parts.append("</div></section>")
    parts.append("</main></body></html>")
    OUTPUT_HTML.write_text("\n".join(parts), encoding="utf-8")

    print(f"OpenAI baseline: {openai_model or 'No valid OpenAI baseline available in this run'}")
    print(f"Self-host candidate: {self_host_model or 'No valid self-host candidate available in this run'}")
    print(f"Saved {OUTPUT_CSV.as_posix()} ({len(compact_rows)} rows)")
    print(f"Saved {OUTPUT_LONG_CSV.as_posix()} ({len(long_rows)} rows)")
    print(f"Saved {OUTPUT_WIDE_CSV.as_posix()} ({len(wide_rows)} rows)")
    print(f"Saved {OUTPUT_HTML.as_posix()}")


if __name__ == "__main__":
    main()
