from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_kik_executive_report import group_results, read_csv, render_report
from vlm_eval.config import ModelConfig, dump_config_snapshot
from vlm_eval.reporting import make_run_dir
from vlm_eval.run import DEFAULT_KIK_REFERENCES, extract_json_object, find_reference_images
from vlm_eval.tasks.kik.data import load_kik_cases, resolve_kik_labels_path
from vlm_eval.tasks.kik.reporting import write_kik_outputs
from vlm_eval.tasks.kik.schema import validate_kik_prediction
from vlm_eval.tasks.kik.scoring import aggregate_kik_by_model, score_kik_fields


DEFAULT_PREDICTIONS = Path("runs/aistudio_gemma4_manual_run/manual_predictions.jsonl")
DEFAULT_OUTPUT_ROOT = Path("runs/kik_eval_gemma4_31b_aistudio_manual")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score JSON answers copied from Google AI Studio against the KIK benchmark."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--images", type=Path, default=Path("data/real_images"))
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--references", type=Path, default=DEFAULT_KIK_REFERENCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-key", type=str, default="aistudio_gemma4_31b")
    parser.add_argument("--model", type=str, default="gemma-4-31b-it")
    parser.add_argument("--provider", type=str, default="google_aistudio_ui")
    parser.add_argument("--strict", action="store_true", help="Fail if any row is skipped or invalid.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels_path, label_mode = resolve_kik_labels_path(args.labels)
    cases = {case.image: case for case in load_kik_cases(args.images, labels_path)}
    references = find_reference_images(args.references)
    rows, skipped = score_manual_predictions(args, cases)

    run_dir = make_run_dir(args.output)
    model = ModelConfig(
        key=args.model_key,
        role="production_candidate",
        canonical_model=args.model,
        provider=args.provider,
        provider_model=args.model,
        enabled_by_default=False,
        heavy=False,
        temperature=0.0,
        max_output_tokens=512,
        image_max_side=1024,
        response_format="json_object",
    )
    dump_config_snapshot(
        run_dir / "config_snapshot.yaml",
        {
            "task": "kik",
            "source": "manual Google AI Studio UI JSON responses",
            "predictions": str(args.predictions),
            "images": str(args.images),
            "labels": str(labels_path),
            "labels_mode": label_mode,
            "models": args.model_key,
            "provider": args.provider,
            "references": ",".join(str(path) for path in references),
            "reference_count": len(references),
            "skipped_empty_or_missing_rows": len(skipped),
            "image_prompt_contract": (
                "AI Studio manual: REF_01..REF_07 uploaded first, then exactly one TARGET_00; "
                "refs are visual dictionary only, target is the only scored photo"
            ),
        },
        [model],
    )

    aggregates = aggregate_kik_by_model(rows)
    write_kik_outputs(
        run_dir,
        rows,
        aggregates,
        dataset_size=len(rows),
        labels_path=labels_path,
        models_tested=[args.model_key],
    )
    html_path = write_html_report(run_dir, args.images)

    error_count = sum(1 for row in rows if row.get("error"))
    print(f"Run dir: {run_dir}")
    print(f"Scored rows: {len(rows)}")
    if skipped:
        print("Skipped empty/missing rows: " + ", ".join(skipped))
    print(f"Rows with errors: {error_count}")
    print(f"HTML: {html_path}")
    if args.strict and (skipped or error_count):
        return 2
    return 0


def score_manual_predictions(
    args: argparse.Namespace,
    cases: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    seen: set[str] = set()
    for line_no, item in read_prediction_rows(args.predictions):
        image = str(item.get("image_id") or item.get("image") or "").strip()
        if not image:
            raise ValueError(f"{args.predictions}:{line_no}: missing image/image_id")
        if image in seen:
            raise ValueError(f"{args.predictions}:{line_no}: duplicate image {image}")
        seen.add(image)
        case = cases.get(image)
        if case is None:
            raise ValueError(f"{args.predictions}:{line_no}: unknown image {image}")

        raw_payload = prediction_payload(item)
        if is_empty_prediction(raw_payload):
            skipped.append(image)
            continue

        row = base_result_row(args, image, case.expected)
        row["raw_response"] = raw_text(raw_payload)
        try:
            parsed = parse_prediction(raw_payload)
        except Exception as exc:
            row["error"] = f"JSON parse error: {exc}"
            rows.append(row)
            continue

        validation = validate_kik_prediction(parsed)
        row.update(
            {
                "json_parse_ok": True,
                "schema_valid": validation.ok,
                "parsed": parsed,
                "field_scores": score_kik_fields(case.expected, parsed),
                "error": None if validation.ok else "; ".join(validation.errors),
            }
        )
        rows.append(row)
    if not rows:
        raise ValueError(f"No non-empty predictions found in {args.predictions}")
    return rows, skipped


def read_prediction_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    text = path.read_text(encoding="utf-8")
    try:
        return expand_prediction_item(path, 1, json.loads(text))
    except json.JSONDecodeError:
        pass

    rows: list[tuple[int, dict[str, Any]]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        rows.extend(expand_prediction_item(path, line_no, item))
    return rows


def expand_prediction_item(path: Path, line_no: int, item: Any) -> list[tuple[int, dict[str, Any]]]:
    if isinstance(item, list):
        return [(line_no, normalize_batch_prediction(path, line_no, obj)) for obj in item]
    if not isinstance(item, dict):
        raise ValueError(f"{path}:{line_no}: each line must be a JSON object, array, or batch wrapper")
    for key in ("predictions", "results", "items"):
        value = item.get(key)
        if isinstance(value, list):
            return [(line_no, normalize_batch_prediction(path, line_no, obj)) for obj in value]
    return [(line_no, item)]


def normalize_batch_prediction(path: Path, line_no: int, item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"{path}:{line_no}: batch item must be a JSON object")
    image = str(item.get("image") or item.get("image_id") or item.get("target_id") or "").strip()
    if image.startswith("TARGET_"):
        image = image.removeprefix("TARGET_00_")
        image = image.removeprefix("TARGET_")
    if image and not image.endswith(".jpg"):
        image = f"{image}.jpg"
    if not image:
        raise ValueError(f"{path}:{line_no}: batch item missing image/image_id/target_id")
    prediction = dict(item)
    prediction.pop("image", None)
    prediction.pop("image_id", None)
    prediction.pop("target_id", None)
    return {"image": image, "raw_response": prediction}


def prediction_payload(item: dict[str, Any]) -> Any:
    if "parsed" in item:
        return item["parsed"]
    if "raw_response" in item:
        return item["raw_response"]
    if "response" in item:
        return item["response"]
    return None


def is_empty_prediction(value: Any) -> bool:
    return value is None or value == "" or value == {}


def parse_prediction(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return extract_json_object(value)
    raise ValueError(f"prediction must be object or string, got {type(value).__name__}")


def raw_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def base_result_row(args: argparse.Namespace, image: str, expected: dict[str, Any]) -> dict[str, Any]:
    return {
        "image": image,
        "task": "kik",
        "model_key": args.model_key,
        "model": args.model,
        "role": "production_candidate",
        "provider": args.provider,
        "provider_model": args.model,
        "latency_ms": None,
        "json_parse_ok": False,
        "schema_valid": False,
        "retry_count": 0,
        "api_retry_count": 0,
        "token_usage": None,
        "raw_response": "",
        "parsed": None,
        "expected": expected,
        "field_scores": {},
        "error": None,
        "response_format_mode": "manual_json",
    }


def write_html_report(run_dir: Path, images_dir: Path) -> Path:
    output = run_dir / "kik_executive_model_report.html"
    summaries = read_csv(run_dir / "summary.csv")
    summaries = sorted(
        summaries,
        key=lambda row: float(row.get("kik_business_score_pct") or -1),
        reverse=True,
    )
    rank_map = {row["model_key"]: index + 1 for index, row in enumerate(summaries)}
    results = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    html_text = render_report(run_dir, images_dir, output, summaries, group_results(results), rank_map)
    output.write_text(html_text, encoding="utf-8")
    return output


if __name__ == "__main__":
    raise SystemExit(main())
