from pathlib import Path

import pandas as pd

GROUND_TRUTH_JSONL = Path("data/ground_truth/manual_ground_truth.jsonl")
PREDICTIONS_CSV = Path("results/openrouter_eval_results.csv")

DETAILS_CSV = Path("results/model_comparison_details.csv")
SUMMARY_CSV = Path("results/model_comparison_summary.csv")
BOOLEAN_BY_MODEL_CSV = Path("results/boolean_metrics_by_model.csv")
NUMERIC_BY_MODEL_CSV = Path("results/numeric_metrics_by_model.csv")
FIELD_COVERAGE_BY_MODEL_CSV = Path("results/field_coverage_by_model.csv")


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
    pred = pd.read_csv(PREDICTIONS_CSV)

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
    for model, group in merged.groupby("model", dropna=False):
        errors_count = int((group["error"].astype(str).str.strip() != "").sum())
        avg_latency = float(group["latency_sec"].mean()) if group["latency_sec"].notna().any() else float("nan")

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
                pred_non_null = int(group[f"{f}_pred"].notna().sum())
                pred_non_null_on_gt = int((group[f"{f}_gt"].notna() & group[f"{f}_pred"].notna()).sum())
                missing_pred_on_gt = int((group[f"{f}_gt"].notna() & group[f"{f}_pred"].isna()).sum())
                numeric_rows.append(
                    {
                        "model": model,
                        "field": f,
                        "mae": round(mae, 4),
                        "rmse": round(rmse, 4),
                        "n": gt_non_null,
                    }
                )
                coverage_rows.append(
                    {
                        "model": model,
                        "field": f,
                        "field_type": "numeric",
                        "gt_non_null": gt_non_null,
                        "pred_non_null_total": pred_non_null,
                        "pred_non_null_on_gt": pred_non_null_on_gt,
                        "missing_pred_when_gt_known": missing_pred_on_gt,
                        "coverage_on_gt": round(pred_non_null_on_gt / gt_non_null, 4) if gt_non_null > 0 else float("nan"),
                    }
                )

        for f in BOOLEAN_FIELDS:
            if f"{f}_gt" not in group.columns or f"{f}_pred" not in group.columns:
                continue
            m = boolean_metrics(group[f"{f}_gt"], group[f"{f}_pred"])
            boolean_rows.append({"model": model, "field": f, **m})
            gt_non_null = int(group[f"{f}_gt"].notna().sum())
            pred_non_null = int(group[f"{f}_pred"].notna().sum())
            pred_non_null_on_gt = int((group[f"{f}_gt"].notna() & group[f"{f}_pred"].notna()).sum())
            missing_pred_on_gt = int((group[f"{f}_gt"].notna() & group[f"{f}_pred"].isna()).sum())
            coverage_rows.append(
                {
                    "model": model,
                    "field": f,
                    "field_type": "boolean",
                    "gt_non_null": gt_non_null,
                    "pred_non_null_total": pred_non_null,
                    "pred_non_null_on_gt": pred_non_null_on_gt,
                    "missing_pred_when_gt_known": missing_pred_on_gt,
                    "coverage_on_gt": round(pred_non_null_on_gt / gt_non_null, 4) if gt_non_null > 0 else float("nan"),
                }
            )

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

    print(f"Saved details: {DETAILS_CSV.as_posix()}")
    print(f"Saved summary: {SUMMARY_CSV.as_posix()}")
    print(f"Saved boolean metrics: {BOOLEAN_BY_MODEL_CSV.as_posix()}")
    print(f"Saved numeric metrics: {NUMERIC_BY_MODEL_CSV.as_posix()}")
    print(f"Saved field coverage: {FIELD_COVERAGE_BY_MODEL_CSV.as_posix()}")


if __name__ == "__main__":
    main()
