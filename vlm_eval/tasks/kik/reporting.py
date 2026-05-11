from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from ...reporting import write_csv, write_jsonl
from .schema import KIK_REQUIRED_FIELDS

KIK_SUMMARY_COLUMNS = [
    "model_key",
    "model",
    "role",
    "total_cases",
    "api_or_json_errors",
    "json_parse_rate",
    "schema_valid_rate",
    "avg_latency_sec",
    "p95_latency_sec",
    "kik_business_score_pct",
    "core_kik_score_pct",
    "sku_family_score_pct",
    "execution_score_pct",
    "equipment_photo_score_pct",
    "status_actionability_score_pct",
    "kik_present_accuracy",
    "kik_present_precision",
    "kik_present_recall",
    "kik_present_f1",
    "kik_false_positive_rate",
    "kik_false_negative_rate",
    "kik_sku_count_mae",
    "sku_within_1_accuracy",
    "sku_within_2_accuracy",
    "kik_share_percent_mae",
    "share_within_10pp_accuracy",
    "sku_family_macro_f1",
    "execution_macro_f1",
    "status_score_accuracy",
    "critical_recall",
    "critical_precision",
    "false_normal_on_critical_count",
    "field_coverage_rate",
    "null_rate_avg",
]


def write_kik_outputs(
    run_dir: Path,
    results: list[dict[str, Any]],
    aggregates: dict[str, Any],
    dataset_size: int,
    labels_path: Path,
    models_tested: list[str],
) -> None:
    errors = [row for row in results if row.get("error")]
    write_jsonl(run_dir / "results.jsonl", sorted(results, key=lambda row: (row["model_key"], row["image"])))
    write_jsonl(run_dir / "errors.jsonl", errors)
    write_csv(run_dir / "summary.csv", aggregates["summaries"], KIK_SUMMARY_COLUMNS)
    write_csv(run_dir / "boolean_metrics_by_model.csv", aggregates["boolean_rows"])
    write_csv(run_dir / "numeric_metrics_by_model.csv", aggregates["numeric_rows"])
    write_csv(run_dir / "business_key_metrics_by_model.csv", aggregates["business_rows"])
    write_csv(run_dir / "field_coverage_by_model.csv", aggregates["coverage_rows"])
    write_csv(run_dir / "worst_cases_by_model.csv", aggregates["worst_rows"])
    write_csv(run_dir / "confusion_status_score.csv", aggregates["confusion_rows"])
    write_summary_md(
        run_dir / "summary.md",
        aggregates["summaries"],
        aggregates["worst_rows"],
        dataset_size,
        labels_path,
        models_tested,
    )


def write_summary_md(
    path: Path,
    summaries: list[dict[str, Any]],
    worst_rows: list[dict[str, Any]],
    dataset_size: int,
    labels_path: Path,
    models_tested: list[str],
) -> None:
    lines = [
        "# KIK Retail Execution Eval Summary",
        "",
        f"- Benchmark date: {datetime.now().isoformat(timespec='seconds')}",
        f"- Dataset size: {dataset_size}",
        f"- GT source: {labels_path.as_posix()}",
        f"- Models tested: {', '.join(models_tested)}",
        "",
        "## Main Ranking",
        "",
        *_table(
            sorted(summaries, key=lambda row: _sort_value(row.get("kik_business_score_pct")), reverse=True),
            [
                "model_key",
                "role",
                "kik_business_score_pct",
                "core_kik_score_pct",
                "sku_family_score_pct",
                "execution_score_pct",
                "critical_recall",
                "schema_valid_rate",
                "p95_latency_sec",
            ],
        ),
        "",
    ]

    role_titles = [
        ("quality_ceiling", "Heavy Benchmark / Quality Ceiling"),
        ("production_candidate", "Self-Host Candidates"),
        ("weak_baseline", "Weak Baseline"),
        ("mock", "Mock"),
    ]
    for role, title in role_titles:
        rows = [row for row in summaries if row.get("role") == role]
        if not rows:
            continue
        lines.extend([f"## {title}", ""])
        lines.extend(
            _table(
                sorted(rows, key=lambda row: _sort_value(row.get("kik_business_score_pct")), reverse=True),
                [
                    "model_key",
                    "kik_business_score_pct",
                    "kik_present_f1",
                    "kik_sku_count_mae",
                    "kik_share_percent_mae",
                    "sku_family_macro_f1",
                    "execution_macro_f1",
                    "critical_recall",
                ],
            )
        )
        lines.append("")

    best_checks = [
        ("KIK presence", "kik_present_f1", False),
        ("SKU count", "kik_sku_count_mae", True),
        ("KIK share", "kik_share_percent_mae", True),
        ("SKU family detection", "sku_family_macro_f1", False),
        ("Execution violations", "execution_macro_f1", False),
        ("Critical status detection", "critical_recall", False),
    ]
    lines.extend(["## Best By Business Need", ""])
    for label, field, lower_is_better in best_checks:
        best = _best_by(summaries, field, lower_is_better=lower_is_better)
        lines.append(f"- {label}: {_best_text(best, field)}")

    lines.extend(["", "## Worst Failure Cases", ""])
    for model_key in sorted({str(row.get("model_key")) for row in worst_rows}):
        rows = [row for row in worst_rows if row.get("model_key") == model_key][:5]
        if not rows:
            continue
        cases = ", ".join(f"{row.get('image')} ({_fmt(row.get('kik_business_score_pct'))})" for row in rows)
        lines.append(f"- `{model_key}`: {cases}")

    lines.extend(["", "## Recommendation", ""])
    best_heavy = _best_by([row for row in summaries if row.get("role") == "quality_ceiling"], "kik_business_score_pct")
    best_prod = _best_by([row for row in summaries if row.get("role") == "production_candidate"], "kik_business_score_pct")
    best_ocr = _best_by([row for row in summaries if row.get("role") == "production_candidate"], "sku_family_macro_f1")
    reject = [row["model_key"] for row in summaries if (row.get("kik_business_score_pct") or 0) < 60]
    lines.append(f"- Best heavy benchmark model: {_best_text(best_heavy, 'kik_business_score_pct')}")
    lines.append(f"- Best self-host production candidate: {_best_text(best_prod, 'kik_business_score_pct')}")
    lines.append(f"- Best OCR/fallback candidate: {_best_text(best_ocr, 'sku_family_macro_f1')}")
    lines.append(f"- Models to reject: {', '.join(reject) if reject else 'none from this run'}")
    lines.append(f"- Self-host candidate close enough to heavy ceiling: {_ceiling_gap_text(best_prod, best_heavy)}")

    lines.extend(["", "## Production Thresholds", ""])
    lines.append("- 90-100%: excellent; 85-90%: strong production candidate; 75-85%: fallback/manual-assist; 60-75%: weak baseline; <60%: reject.")
    lines.append("- Hard minimum: kik_present_f1 >= 0.95, kik_sku_count_mae <= 1.5, kik_share_percent_mae <= 10, sku_family_macro_f1 >= 0.85, critical_recall >= 0.90, schema_valid_rate >= 0.98.")

    lines.extend(["", "## KIK Fields", ""])
    lines.append(", ".join(f"`{field}`" for field in KIK_REQUIRED_FIELDS))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |")
    return lines


def _best_by(rows: list[dict[str, Any]], field: str, lower_is_better: bool = False) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get(field) is not None]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: row[field], reverse=not lower_is_better)[0]


def _best_text(row: dict[str, Any] | None, field: str) -> str:
    if not row:
        return "n/a"
    return f"`{row['model_key']}` ({field}={_fmt(row.get(field))})"


def _ceiling_gap_text(prod: dict[str, Any] | None, heavy: dict[str, Any] | None) -> str:
    if not prod or not heavy:
        return "not enough data"
    gap = (heavy.get("kik_business_score_pct") or 0) - (prod.get("kik_business_score_pct") or 0)
    return "yes" if gap <= 5 else f"needs work, gap {_fmt(gap)} pct points"


def _sort_value(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else -1.0


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
