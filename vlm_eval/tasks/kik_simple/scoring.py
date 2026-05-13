from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from .schema import KIK_SIMPLE_BOOLEAN_FIELDS, KIK_SIMPLE_NUMERIC_FIELDS, KIK_SIMPLE_REQUIRED_FIELDS

EXECUTION_FIELDS = [
    "has_monobrand_block",
    "is_kik_mixed_with_competitors",
    "has_non_icecream_products",
]

BUSINESS_WEIGHTS = {
    "kik_present": 16.0,
    "kik_sku_count": 24.0,
    "kik_share_percent": 24.0,
    "has_monobrand_block": 4.0,
    "is_kik_mixed_with_competitors": 4.0,
    "has_non_icecream_products": 1.5,
    "is_trade_equipment_photo": 2.0,
    "is_ice_cream_equipment": 2.0,
    "status_score": 8.0,
}

FIELD_GROUPS = {
    "core_kik_score_pct": {"kik_present", "kik_sku_count", "kik_share_percent"},
    "execution_score_pct": set(EXECUTION_FIELDS),
    "equipment_photo_score_pct": {
        "is_trade_equipment_photo",
        "is_ice_cream_equipment",
    },
    "status_actionability_score_pct": {"status_score"},
}


def score_kik_simple_fields(expected: dict[str, Any], predicted: dict[str, Any] | None) -> dict[str, float]:
    if predicted is None:
        return {field: 0.0 for field, value in expected.items() if _is_scorable_expected(field, value)}
    scores: dict[str, float] = {}
    for field, expected_value in expected.items():
        if not _is_scorable_expected(field, expected_value):
            continue
        scores[field] = score_kik_simple_value(field, expected_value, predicted.get(field))
    return scores


def score_kik_simple_value(field: str, expected: Any, predicted: Any) -> float:
    if field in KIK_SIMPLE_BOOLEAN_FIELDS:
        return 1.0 if isinstance(predicted, bool) and predicted == expected else 0.0
    if field == "kik_sku_count":
        return _bounded_numeric_score(expected, predicted, cap=10)
    if field == "kik_share_percent":
        return _bounded_numeric_score(expected, predicted, cap=50)
    if field == "status_score":
        return 1.0 if _as_int(predicted) == _as_int(expected) else 0.0
    return 0.0


def business_scores(expected: dict[str, Any], predicted: dict[str, Any] | None) -> dict[str, float | None]:
    if _missed_existing_kik_skus(expected, predicted):
        return {
            "kik_simple_business_score_pct": 0.0,
            **{group_name: 0.0 for group_name in FIELD_GROUPS},
        }
    field_scores = score_kik_simple_fields(expected, predicted)
    result = {"kik_simple_business_score_pct": _weighted_pct(field_scores, set(BUSINESS_WEIGHTS))}
    for group_name, fields in FIELD_GROUPS.items():
        result[group_name] = _weighted_pct(field_scores, fields)
    return result


def aggregate_kik_simple_by_model(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["model_key"]].append(result)

    summaries: list[dict[str, Any]] = []
    boolean_rows: list[dict[str, Any]] = []
    numeric_rows: list[dict[str, Any]] = []
    business_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    worst_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []

    for model_key, rows in sorted(grouped.items()):
        meta = rows[0]
        latencies_sec = [
            float(row["latency_ms"]) / 1000.0 for row in rows if isinstance(row.get("latency_ms"), (int, float))
        ]
        parse_ok = _rate([bool(row.get("json_parse_ok")) for row in rows])
        schema_ok = _rate([bool(row.get("schema_valid")) for row in rows])
        errors = [row for row in rows if row.get("error")]
        row_business_scores = [business_scores(row.get("expected") or {}, _parsed(row)) for row in rows]

        summary: dict[str, Any] = {
            "model_key": model_key,
            "model": meta.get("model") or meta.get("provider_model"),
            "role": meta.get("role"),
            "total_cases": len(rows),
            "api_or_json_errors": len(errors),
            "json_parse_rate": parse_ok,
            "schema_valid_rate": schema_ok,
            "avg_latency_sec": mean(latencies_sec) if latencies_sec else None,
            "p50_latency_sec": percentile(latencies_sec, 50),
            "p95_latency_sec": percentile(latencies_sec, 95),
            "retry_rate": _rate([bool(row.get("retry_count")) for row in rows]),
            "api_error_rate": _rate([bool(row.get("error")) and not row.get("json_parse_ok") for row in rows]),
            "total_input_tokens": _sum_usage(rows, "input_tokens"),
            "total_output_tokens": _sum_usage(rows, "output_tokens"),
            "total_tokens": _sum_usage(rows, "total_tokens"),
        }
        for field in [
            "kik_simple_business_score_pct",
            "core_kik_score_pct",
            "execution_score_pct",
            "equipment_photo_score_pct",
            "status_actionability_score_pct",
        ]:
            summary[field] = _clean_mean([score[field] for score in row_business_scores])

        boolean_by_field: dict[str, dict[str, Any]] = {}
        for field in KIK_SIMPLE_BOOLEAN_FIELDS:
            metrics = boolean_metrics(rows, field)
            boolean_by_field[field] = metrics
            boolean_rows.append(_metric_row(meta, field, "boolean", metrics))
        kik_present = boolean_by_field["kik_present"]
        summary.update(
            {
                "kik_present_accuracy": kik_present["accuracy"],
                "kik_present_precision": kik_present["precision"],
                "kik_present_recall": kik_present["recall"],
                "kik_present_f1": kik_present["f1"],
            }
        )
        hallucination = kik_hallucination_metrics(rows)
        summary.update(hallucination)

        numeric_by_field = {field: numeric_metrics(rows, field) for field in KIK_SIMPLE_NUMERIC_FIELDS}
        for field, metrics in numeric_by_field.items():
            numeric_rows.append(_metric_row(meta, field, "numeric", metrics))
        summary.update(
            {
                "kik_sku_count_mae": numeric_by_field["kik_sku_count"]["mae"],
                "sku_within_1_accuracy": numeric_by_field["kik_sku_count"]["within_1_accuracy"],
                "sku_within_2_accuracy": numeric_by_field["kik_sku_count"]["within_2_accuracy"],
                "kik_share_percent_mae": numeric_by_field["kik_share_percent"]["mae"],
                "share_within_10pp_accuracy": numeric_by_field["kik_share_percent"]["within_10_accuracy"],
            }
        )

        summary["execution_macro_f1"] = macro_f1(boolean_by_field, EXECUTION_FIELDS)
        status = status_metrics(rows)
        summary.update(status)

        coverage = coverage_metrics(rows)
        summary.update(
            {
                "field_coverage_rate": coverage["field_coverage_rate"],
                "null_rate_avg": coverage["null_rate_avg"],
            }
        )
        coverage_rows.extend(_coverage_rows(meta, coverage["fields"]))

        business_rows.extend(_business_rows(meta, summary, hallucination))
        worst_rows.extend(worst_case_rows(rows))
        confusion_rows.extend(confusion_status_rows(rows, meta))
        summaries.append(summary)

    return {
        "summaries": summaries,
        "boolean_rows": boolean_rows,
        "numeric_rows": numeric_rows,
        "business_rows": business_rows,
        "coverage_rows": coverage_rows,
        "worst_rows": sorted(worst_rows, key=lambda row: (row["model_key"], row["kik_simple_business_score_pct"])),
        "confusion_rows": confusion_rows,
    }


def boolean_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = []
    for row in rows:
        expected = (row.get("expected") or {}).get(field)
        if expected is None:
            continue
        predicted = _pred_value(row, field)
        values.append((expected, predicted))
    if not values:
        return {"n": 0, "accuracy": None, "precision": None, "recall": None, "f1": None, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    tp = sum(1 for gt, pred in values if gt is True and pred is True)
    fp = sum(1 for gt, pred in values if gt is False and pred is True)
    fn = sum(1 for gt, pred in values if gt is True and pred is not True)
    tn = sum(1 for gt, pred in values if gt is False and pred is False)
    accuracy = sum(1 for gt, pred in values if pred is not None and pred == gt) / len(values)
    precision = tp / (tp + fp) if tp + fp else (0.0 if fn else None)
    recall = tp / (tp + fn) if tp + fn else (0.0 if fp else None)
    if precision is None and recall is None:
        f1 = None
    elif precision is not None and recall is not None and precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {"n": len(values), "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def numeric_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    pairs: list[tuple[int, int | None]] = []
    scores: list[float] = []
    for row in rows:
        expected = _as_int((row.get("expected") or {}).get(field))
        if expected is None:
            continue
        predicted = _as_int(_pred_value(row, field))
        pairs.append((expected, predicted))
        scores.append(score_kik_simple_value(field, expected, predicted))
    errors = [abs(gt - pred) for gt, pred in pairs if pred is not None]
    metrics = {
        "n": len(pairs),
        "mae": mean(errors) if errors else None,
        "normalized_score": mean(scores) if scores else None,
    }
    for tolerance in (1, 2, 5, 10):
        metrics[f"within_{tolerance}_accuracy"] = _within_accuracy(pairs, tolerance)
    return metrics


def status_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = []
    for row in rows:
        expected = _as_int((row.get("expected") or {}).get("status_score"))
        if expected is None:
            continue
        pairs.append((expected, _as_int(_pred_value(row, "status_score"))))
    if not pairs:
        return {
            "status_score_accuracy": None,
            "critical_recall": None,
            "critical_precision": None,
            "false_normal_on_critical_count": 0,
        }
    accuracy = sum(1 for gt, pred in pairs if pred == gt) / len(pairs)
    gt_critical = [(gt, pred) for gt, pred in pairs if gt == 2]
    pred_critical = [(gt, pred) for gt, pred in pairs if pred == 2]
    critical_recall = sum(1 for _, pred in gt_critical if pred == 2) / len(gt_critical) if gt_critical else None
    critical_precision = sum(1 for gt, _ in pred_critical if gt == 2) / len(pred_critical) if pred_critical else None
    false_normal = sum(1 for gt, pred in pairs if gt == 2 and pred == 0)
    return {
        "status_score_accuracy": accuracy,
        "critical_recall": critical_recall,
        "critical_precision": critical_precision,
        "false_normal_on_critical_count": false_normal,
    }


def kik_hallucination_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gt_false = [row for row in rows if (row.get("expected") or {}).get("kik_present") is False]
    gt_true = [row for row in rows if (row.get("expected") or {}).get("kik_present") is True]
    false_positive = sum(1 for row in gt_false if _pred_value(row, "kik_present") is True)
    false_negative = sum(1 for row in gt_true if _pred_value(row, "kik_present") is not True)
    sku_hallucination = sum(1 for row in gt_false if (_as_int(_pred_value(row, "kik_sku_count")) or 0) > 0)
    share_hallucination = sum(1 for row in gt_false if (_as_int(_pred_value(row, "kik_share_percent")) or 0) > 0)
    return {
        "kik_false_positive_rate": false_positive / len(gt_false) if gt_false else None,
        "kik_false_negative_rate": false_negative / len(gt_true) if gt_true else None,
        "sku_hallucination_on_absent_kik": sku_hallucination / len(gt_false) if gt_false else None,
        "share_hallucination_on_absent_kik": share_hallucination / len(gt_false) if gt_false else None,
    }


def coverage_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    field_rows: list[dict[str, Any]] = []
    total_expected = 0
    total_covered = 0
    null_rates: list[float] = []
    for field in KIK_SIMPLE_REQUIRED_FIELDS:
        expected_non_null = 0
        covered = 0
        pred_null = 0
        parsed_rows = 0
        for row in rows:
            parsed = _parsed(row)
            if parsed is not None:
                parsed_rows += 1
                if parsed.get(field) is None:
                    pred_null += 1
            if (row.get("expected") or {}).get(field) is not None:
                expected_non_null += 1
                if parsed is not None and parsed.get(field) is not None:
                    covered += 1
        total_expected += expected_non_null
        total_covered += covered
        null_rate = pred_null / parsed_rows if parsed_rows else None
        if null_rate is not None:
            null_rates.append(null_rate)
        field_rows.append(
            {
                "field": field,
                "expected_non_null_count": expected_non_null,
                "covered_count": covered,
                "field_coverage_rate": covered / expected_non_null if expected_non_null else None,
                "null_rate": null_rate,
            }
        )
    return {
        "field_coverage_rate": total_covered / total_expected if total_expected else None,
        "null_rate_avg": mean(null_rates) if null_rates else None,
        "fields": field_rows,
    }


def worst_case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    worst = []
    for row in rows:
        scores = business_scores(row.get("expected") or {}, _parsed(row))
        field_scores = row.get("field_scores") or {}
        error_fields = [field for field, score in sorted(field_scores.items()) if score < 1.0]
        worst.append(
            {
                "model_key": row["model_key"],
                "model": row.get("model") or row.get("provider_model"),
                "role": row.get("role"),
                "image": row.get("image"),
                "kik_simple_business_score_pct": scores["kik_simple_business_score_pct"],
                "core_kik_score_pct": scores["core_kik_score_pct"],
                "error_field_count": len(error_fields),
                "error_fields": ";".join(error_fields),
                "error": row.get("error"),
            }
        )
    return worst


def confusion_status_rows(rows: list[dict[str, Any]], meta: dict[str, Any]) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, str]] = Counter()
    for row in rows:
        expected = (row.get("expected") or {}).get("status_score")
        if expected is None:
            continue
        predicted = _pred_value(row, "status_score")
        counter[(str(expected), str(predicted))] += 1
    return [
        {
            "model_key": meta.get("model_key"),
            "model": meta.get("model") or meta.get("provider_model"),
            "role": meta.get("role"),
            "expected_status_score": expected,
            "predicted_status_score": predicted,
            "count": count,
        }
        for (expected, predicted), count in sorted(counter.items())
    ]


def macro_f1(metrics_by_field: dict[str, dict[str, Any]], fields: list[str]) -> float | None:
    values = [metrics_by_field[field]["f1"] for field in fields if metrics_by_field.get(field, {}).get("f1") is not None]
    return mean(values) if values else None


def percentile(values: list[float], p: int) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (p / 100)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _business_rows(meta: dict[str, Any], summary: dict[str, Any], hallucination: dict[str, Any]) -> list[dict[str, Any]]:
    fields = [
        "kik_simple_business_score_pct",
        "core_kik_score_pct",
        "execution_score_pct",
        "equipment_photo_score_pct",
        "status_actionability_score_pct",
        "execution_macro_f1",
        "critical_recall",
        "critical_precision",
        "false_normal_on_critical_count",
    ]
    rows = [
        {
            "model_key": meta.get("model_key"),
            "model": meta.get("model") or meta.get("provider_model"),
            "role": meta.get("role"),
            "metric": field,
            "value": summary.get(field),
        }
        for field in fields
    ]
    rows.extend(
        {
            "model_key": meta.get("model_key"),
            "model": meta.get("model") or meta.get("provider_model"),
            "role": meta.get("role"),
            "metric": field,
            "value": value,
        }
        for field, value in hallucination.items()
    )
    return rows


def _coverage_rows(meta: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "model_key": meta.get("model_key"),
            "model": meta.get("model") or meta.get("provider_model"),
            "role": meta.get("role"),
            **row,
        }
        for row in rows
    ]


def _metric_row(meta: dict[str, Any], field: str, field_type: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_key": meta.get("model_key"),
        "model": meta.get("model") or meta.get("provider_model"),
        "role": meta.get("role"),
        "field": field,
        "field_type": field_type,
        **metrics,
    }


def _weighted_pct(field_scores: dict[str, float], fields: set[str]) -> float | None:
    weighted_sum = 0.0
    weight_sum = 0.0
    for field in fields:
        if field not in field_scores:
            continue
        weight = BUSINESS_WEIGHTS[field]
        weighted_sum += weight * field_scores[field]
        weight_sum += weight
    if not weight_sum:
        return None
    return 100.0 * weighted_sum / weight_sum


def _bounded_numeric_score(expected: Any, predicted: Any, cap: int) -> float:
    gt = _as_int(expected)
    pred = _as_int(predicted)
    if gt is None or pred is None:
        return 0.0
    return max(0.0, 1.0 - min(abs(pred - gt), cap) / cap)


def _within_accuracy(pairs: list[tuple[int, int | None]], tolerance: int) -> float | None:
    if not pairs:
        return None
    return sum(1 for gt, pred in pairs if pred is not None and abs(gt - pred) <= tolerance) / len(pairs)


def _is_scorable_expected(field: str, value: Any) -> bool:
    return field in BUSINESS_WEIGHTS and value is not None


def _missed_existing_kik_skus(expected: dict[str, Any], predicted: dict[str, Any] | None) -> bool:
    if predicted is None:
        return False
    expected_sku_count = _as_int(expected.get("kik_sku_count"))
    predicted_sku_count = _as_int(predicted.get("kik_sku_count"))
    return expected_sku_count is not None and expected_sku_count > 0 and predicted_sku_count == 0


def _parsed(row: dict[str, Any]) -> dict[str, Any] | None:
    parsed = row.get("parsed")
    return parsed if isinstance(parsed, dict) else None


def _pred_value(row: dict[str, Any], field: str) -> Any:
    parsed = _parsed(row)
    if parsed is None:
        return None
    return parsed.get(field)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rate(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _clean_mean(values: list[Any]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return mean(clean) if clean else None


def _sum_usage(rows: list[dict[str, Any]], key: str) -> int | None:
    values: list[int] = []
    for row in rows:
        usage = row.get("token_usage")
        if not isinstance(usage, dict):
            continue
        value = usage.get(key)
        if isinstance(value, (int, float)):
            values.append(int(value))
    return sum(values) if values else None
