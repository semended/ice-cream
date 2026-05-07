from pathlib import Path

import pandas as pd

GROUND_TRUTH_JSONL = Path("data/ground_truth/manual_ground_truth.jsonl")
PREDICTIONS_CSV = Path("results/openrouter_eval_results.csv")

DETAILS_CSV = Path("results/model_comparison_details.csv")
SUMMARY_CSV = Path("results/model_comparison_summary.csv")
BOOLEAN_BY_MODEL_CSV = Path("results/boolean_metrics_by_model.csv")
NUMERIC_BY_MODEL_CSV = Path("results/numeric_metrics_by_model.csv")
FIELD_COVERAGE_BY_MODEL_CSV = Path("results/field_coverage_by_model.csv")
BUSINESS_KEY_METRICS_BY_MODEL_CSV = Path("results/business_key_metrics_by_model.csv")
WORST_CASES_BY_MODEL_CSV = Path("results/worst_cases_by_model.csv")


NUMERIC_FIELDS = [
    "photo_quality_score",
    "analysis_possible_score",
    "kik_sku_count",
    "kik_share_percent",
    "fill_level_percent",
    "status_score",
    "confidence_score",
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
    "has_monobrand_block",
    "has_foreign_label",
    "has_non_icecream_products",
    "has_empty_sections",
    "is_kik_mixed_with_competitors",
]

NORMALIZERS = {
    "photo_quality_score": 2,
    "analysis_possible_score": 2,
    "kik_sku_count": 15,
    "kik_share_percent": 100,
    "fill_level_percent": 100,
    "status_score": 2,
    "confidence_score": 2,
}

BUSINESS_KEY_FIELDS = [
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
    "has_posm",
    "has_monobrand_block",
    "has_foreign_label",
    "has_non_icecream_products",
    "has_empty_sections",
    "is_kik_mixed_with_competitors",
    "status_score",
]


def _norm_str(s: object) -> str:
    if s is None:
        return ""
    if isinstance(s, float) and pd.isna(s):
        return ""
    return str(s).strip()


def _is_unknown_or_null(x: object) -> bool:
    s = _norm_str(x).lower()
    return s in ("", "unknown", "null", "none", "nan", "-")


def parse_bool_series(series: pd.Series) -> pd.Series:
    def _parse(x: object) -> bool | None:
        if isinstance(x, bool):
            return x
        s = _norm_str(x).lower()
        if s in ("", "unknown", "null", "none", "nan", "-"):
            return None
        if s in ("true", "1", "yes", "y", "да", "истина"):
            return True
        if s in ("false", "0", "no", "n", "нет", "ложь"):
            return False
        return None

    return series.map(_parse)


def parse_int_series(series: pd.Series) -> pd.Series:
    def _parse(x: object) -> int | None:
        if _is_unknown_or_null(x):
            return None
        s = _norm_str(x)
        try:
            return int(float(s))
        except Exception:  # noqa: BLE001
            return None

    return series.map(_parse)


def safe_mae(series_true: pd.Series, series_pred: pd.Series) -> float:
    true_num = pd.to_numeric(series_true, errors="coerce")
    pred_num = pd.to_numeric(series_pred, errors="coerce")
    valid_gt = true_num.notna()
    if valid_gt.sum() == 0:
        return float("nan")
    errors = (true_num - pred_num).abs()
    return errors[valid_gt].mean()


def safe_rmse(series_true: pd.Series, series_pred: pd.Series) -> float:
    true_num = pd.to_numeric(series_true, errors="coerce")
    pred_num = pd.to_numeric(series_pred, errors="coerce")
    valid_gt = true_num.notna()
    if valid_gt.sum() == 0:
        return float("nan")
    sq_err = (true_num - pred_num) ** 2
    return (sq_err[valid_gt].mean()) ** 0.5


def safe_accuracy(series_true: pd.Series, series_pred: pd.Series) -> float:
    valid_gt = series_true.notna()
    if valid_gt.sum() == 0:
        return float("nan")
    correct = (series_true == series_pred) & series_pred.notna()
    return (correct[valid_gt]).mean()


def boolean_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    valid = y_true.notna()
    if valid.sum() == 0:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }

    yt = y_true[valid]
    yp = y_pred[valid]

    tp = int(((yt == True) & (yp == True)).sum())  # noqa: E712
    fp = int(((yt == False) & (yp == True)).sum())  # noqa: E712
    tn = int(((yt == False) & (yp == False)).sum())  # noqa: E712
    fn = int(((yt == True) & (yp != True)).sum())  # noqa: E712

    acc = float(((yt == yp) & yp.notna()).mean())

    prec_den = tp + fp
    rec_den = tp + fn
    precision = float(tp / prec_den) if prec_den > 0 else float("nan")
    recall = float(tp / rec_den) if rec_den > 0 else float("nan")
    if precision == precision and recall == recall and (precision + recall) > 0:
        f1 = float(2 * precision * recall / (precision + recall))
    else:
        f1 = float("nan")

    return {
        "n": int(valid.sum()),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def finite_mean(values: list[float]) -> float:
    clean = [float(v) for v in values if pd.notna(v)]
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


def clamp01(value: float) -> float:
    if pd.isna(value):
        return float("nan")
    return float(max(0.0, min(1.0, value)))


def round_or_nan(value: float, digits: int = 4) -> float:
    return round(float(value), digits) if pd.notna(value) else float("nan")


def coverage_metrics(group: pd.DataFrame, field: str, field_type: str) -> dict[str, object]:
    gt_col = f"{field}_gt"
    pred_col = f"{field}_pred"
    total_cases = int(len(group))
    gt_non_null_count = int(group[gt_col].notna().sum())
    pred_non_null_count = int(group[pred_col].notna().sum())
    both_non_null_count = int((group[gt_col].notna() & group[pred_col].notna()).sum())
    pred_null_when_gt_exists_count = int((group[gt_col].notna() & group[pred_col].isna()).sum())
    gt_coverage = both_non_null_count / gt_non_null_count if gt_non_null_count > 0 else float("nan")
    pred_coverage = pred_non_null_count / total_cases if total_cases > 0 else float("nan")

    return {
        "field": field,
        "field_type": field_type,
        "total_cases": total_cases,
        "gt_non_null_count": gt_non_null_count,
        "pred_non_null_count": pred_non_null_count,
        "both_non_null_count": both_non_null_count,
        "gt_coverage_percent": round_or_nan(gt_coverage * 100),
        "pred_coverage_percent": round_or_nan(pred_coverage * 100),
        "pred_null_when_gt_exists_count": pred_null_when_gt_exists_count,
        "coverage_on_gt": round_or_nan(gt_coverage),
    }


def numeric_error_score(gt_value: object, pred_value: object, field: str) -> float:
    if pd.isna(gt_value):
        return 0.0
    if pd.isna(pred_value):
        return 0.5
    normalizer = NORMALIZERS.get(field, 1)
    if normalizer <= 0:
        normalizer = 1
    return abs(float(gt_value) - float(pred_value)) / normalizer


def worst_case_row(row: pd.Series) -> dict[str, object]:
    error_score = 0.0
    error_fields: list[str] = []

    for field in BOOLEAN_FIELDS:
        gt_value = row.get(f"{field}_gt")
        pred_value = row.get(f"{field}_pred")
        if pd.isna(gt_value):
            continue
        if pd.isna(pred_value):
            error_score += 0.5
            error_fields.append(f"{field}:missing")
        elif bool(gt_value) != bool(pred_value):
            error_score += 1.0
            error_fields.append(f"{field}:wrong")

    for field in NUMERIC_FIELDS:
        gt_value = row.get(f"{field}_gt")
        pred_value = row.get(f"{field}_pred")
        if pd.isna(gt_value):
            continue
        if pd.isna(pred_value):
            error_score += 0.5
            error_fields.append(f"{field}:missing")
            continue
        field_error = numeric_error_score(gt_value, pred_value, field)
        if field_error:
            error_score += field_error
            error_fields.append(f"{field}:{round(field_error, 4)}")

    return {
        "model": row.get("model"),
        "image_id": row.get("image_id"),
        "error_score": round(error_score, 4),
        "error": row.get("error", ""),
        "error_fields": ";".join(error_fields),
        "prediction_json": row.get("prediction_json", ""),
    }


def load_ground_truth() -> pd.DataFrame:
    if not GROUND_TRUTH_JSONL.exists():
        raise FileNotFoundError(f"Ground truth JSONL not found: {GROUND_TRUTH_JSONL.as_posix()}")
    gt = pd.read_json(GROUND_TRUTH_JSONL, lines=True)

    gt["image_id"] = gt["image_id"].astype(str)

    for f in BOOLEAN_FIELDS:
        if f in gt.columns:
            gt[f] = parse_bool_series(gt[f])
    for f in NUMERIC_FIELDS:
        if f in gt.columns:
            gt[f] = parse_int_series(gt[f])

    return gt


def main() -> None:
    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(f"Predictions file not found: {PREDICTIONS_CSV}")

    gt = load_ground_truth()
    pred = pd.read_csv(PREDICTIONS_CSV, keep_default_na=False)

    gt["image_id"] = gt["image_id"].astype(str)
    pred["image_id"] = pred["image_id"].astype(str)

    for f in BOOLEAN_FIELDS:
        if f in pred.columns:
            pred[f] = parse_bool_series(pred[f])
    for f in NUMERIC_FIELDS:
        if f in pred.columns:
            pred[f] = parse_int_series(pred[f])

    pred["latency_sec"] = pd.to_numeric(pred.get("latency_sec"), errors="coerce")
    pred["error"] = pred.get("error", "").astype(str)

    merged = pred.merge(gt, on="image_id", how="left", suffixes=("_pred", "_gt"))

    DETAILS_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(DETAILS_CSV, index=False, encoding="utf-8-sig")

    summary_rows = []
    boolean_rows: list[dict[str, object]] = []
    numeric_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    business_rows: list[dict[str, object]] = []
    worst_case_rows: list[dict[str, object]] = []
    for model, group in merged.groupby("model", dropna=False):
        errors_count = int((group["error"].astype(str).str.strip() != "").sum())
        avg_latency = float(group["latency_sec"].mean()) if group["latency_sec"].notna().any() else float("nan")
        boolean_accuracies: list[float] = []
        numeric_maes: list[float] = []
        normalized_maes: list[float] = []
        coverage_both_non_null = 0
        coverage_gt_non_null = 0

        row: dict[str, object] = {
            "model": model,
            "total_cases": int(len(group)),
            "api_or_json_errors": errors_count,
            "avg_latency_sec": round(avg_latency, 3) if avg_latency == avg_latency else float("nan"),
        }

        for f in NUMERIC_FIELDS:
            if f"{f}_gt" in group.columns and f"{f}_pred" in group.columns:
                mae = safe_mae(group[f"{f}_gt"], group[f"{f}_pred"])
                rmse = safe_rmse(group[f"{f}_gt"], group[f"{f}_pred"])
                row[f"mae_{f}"] = round(mae, 3)
                gt_non_null = int(group[f"{f}_gt"].notna().sum())
                if gt_non_null > 0 and pd.notna(mae):
                    numeric_maes.append(float(mae))
                    normalized_maes.append(min(1.0, float(mae) / NORMALIZERS[f]))
                numeric_rows.append(
                    {
                        "model": model,
                        "field": f,
                        "mae": round(mae, 4),
                        "rmse": round(rmse, 4),
                        "n": gt_non_null,
                    }
                )
                coverage = coverage_metrics(group, f, "numeric")
                coverage_rows.append({"model": model, **coverage})
                coverage_gt_non_null += int(coverage["gt_non_null_count"])
                coverage_both_non_null += int(coverage["both_non_null_count"])

                if f in BUSINESS_KEY_FIELDS:
                    business_rows.append(
                        {
                            "model": model,
                            "field": f,
                            "field_type": "numeric",
                            "mae": round(mae, 4),
                            "rmse": round(rmse, 4),
                            "n": gt_non_null,
                        }
                    )

        for f in BOOLEAN_FIELDS:
            if f"{f}_gt" not in group.columns or f"{f}_pred" not in group.columns:
                continue
            m = boolean_metrics(group[f"{f}_gt"], group[f"{f}_pred"])
            boolean_rows.append({"model": model, "field": f, **m})
            if int(m["n"]) > 0 and pd.notna(m["accuracy"]):
                boolean_accuracies.append(float(m["accuracy"]))
            coverage = coverage_metrics(group, f, "boolean")
            coverage_rows.append({"model": model, **coverage})
            coverage_gt_non_null += int(coverage["gt_non_null_count"])
            coverage_both_non_null += int(coverage["both_non_null_count"])

            if f in BUSINESS_KEY_FIELDS:
                business_rows.append(
                    {
                        "model": model,
                        "field": f,
                        "field_type": "boolean",
                        **m,
                    }
                )

        boolean_macro_accuracy = finite_mean(boolean_accuracies)
        numeric_macro_mae = finite_mean(numeric_maes)
        normalized_mae = finite_mean(normalized_maes)
        numeric_score = clamp01(1.0 - normalized_mae)
        coverage_score = (
            coverage_both_non_null / coverage_gt_non_null if coverage_gt_non_null > 0 else float("nan")
        )
        mvp_parts = [boolean_macro_accuracy, numeric_score, coverage_score]
        if all(pd.notna(part) for part in mvp_parts):
            mvp_score = 0.45 * boolean_macro_accuracy + 0.35 * numeric_score + 0.20 * coverage_score
        else:
            mvp_score = float("nan")

        row["boolean_macro_accuracy"] = round_or_nan(boolean_macro_accuracy)
        row["numeric_macro_mae"] = round_or_nan(numeric_macro_mae)
        row["normalized_numeric_mae"] = round_or_nan(normalized_mae)
        row["numeric_score"] = round_or_nan(numeric_score)
        row["coverage_score"] = round_or_nan(coverage_score)
        row["mvp_score"] = round_or_nan(mvp_score)

        for _, case_row in group.iterrows():
            worst_case_rows.append(worst_case_row(case_row))

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    boolean_df = pd.DataFrame(boolean_rows)
    boolean_df.to_csv(BOOLEAN_BY_MODEL_CSV, index=False, encoding="utf-8-sig")
    numeric_df = pd.DataFrame(numeric_rows)
    numeric_df.to_csv(NUMERIC_BY_MODEL_CSV, index=False, encoding="utf-8-sig")
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(FIELD_COVERAGE_BY_MODEL_CSV, index=False, encoding="utf-8-sig")
    business_df = pd.DataFrame(business_rows)
    business_df.to_csv(BUSINESS_KEY_METRICS_BY_MODEL_CSV, index=False, encoding="utf-8-sig")
    worst_cases_df = pd.DataFrame(worst_case_rows)
    worst_cases_df = worst_cases_df.sort_values(["model", "error_score"], ascending=[True, False])
    worst_cases_df.to_csv(WORST_CASES_BY_MODEL_CSV, index=False, encoding="utf-8-sig")

    print(f"Saved details: {DETAILS_CSV.as_posix()}")
    print(f"Saved summary: {SUMMARY_CSV.as_posix()}")
    print(f"Saved boolean metrics: {BOOLEAN_BY_MODEL_CSV.as_posix()}")
    print(f"Saved numeric metrics: {NUMERIC_BY_MODEL_CSV.as_posix()}")
    print(f"Saved field coverage: {FIELD_COVERAGE_BY_MODEL_CSV.as_posix()}")
    print(f"Saved business key metrics: {BUSINESS_KEY_METRICS_BY_MODEL_CSV.as_posix()}")
    print(f"Saved worst cases: {WORST_CASES_BY_MODEL_CSV.as_posix()}")


if __name__ == "__main__":
    main()
