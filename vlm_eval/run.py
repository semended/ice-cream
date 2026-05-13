from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import ModelConfig, dump_config_snapshot, load_models, select_models
from .data import EvalCase
from .image_utils import encode_image_data_url
from .providers import ProviderError, create_provider, response_format_candidates
from .reporting import make_run_dir
from .tasks.kik.data import load_kik_cases, resolve_kik_labels_path
from .tasks.kik.prompts import (
    SYSTEM_PROMPT as KIK_SYSTEM_PROMPT,
    USER_PROMPT as KIK_USER_PROMPT,
    json_schema_instruction as kik_json_schema_instruction,
)
from .tasks.kik.reporting import write_kik_outputs
from .tasks.kik.schema import (
    KIK_REQUIRED_FIELDS,
    make_mock_kik_prediction,
    response_format_json_schema as kik_response_format_json_schema,
    validate_kik_prediction,
)
from .tasks.kik.scoring import aggregate_kik_by_model, score_kik_fields
from .tasks.kik_simple.data import (
    load_kik_simple_cases,
    resolve_kik_simple_labels_path,
)
from .tasks.kik_simple.prompts import (
    SYSTEM_PROMPT as KIK_SIMPLE_SYSTEM_PROMPT,
    USER_PROMPT as KIK_SIMPLE_USER_PROMPT,
    image_prompt_contract as kik_simple_image_prompt_contract,
    json_schema_instruction as kik_simple_json_schema_instruction,
)
from .tasks.kik_simple.reporting import write_kik_simple_outputs
from .tasks.kik_simple.schema import (
    KIK_SIMPLE_REQUIRED_FIELDS,
    make_mock_kik_simple_prediction,
    response_format_json_schema as kik_simple_response_format_json_schema,
    validate_kik_simple_prediction,
)
from .tasks.kik_simple.scoring import aggregate_kik_simple_by_model, score_kik_simple_fields

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_KIK_REFERENCES = Path("data/reference_images_slides")
DEFAULT_KIK_SIMPLE_REFERENCE = Path("data/reference_images_sku_sheet/kik_sku_reference.png")

REFERENCE_ORDER = {
    "ref_briquette": 1,
    "ref_bucket": 2,
    "ref_cone": 3,
    "ref_cups": 4,
    "ref_eskimo": 5,
    "ref_lakomka": 6,
    "ref_sandwich": 7,
}

REFERENCE_LABELS = {
    "ref_briquette": "REF_01 = KIK briquette / log / brick examples -> JSON field has_poleno_or_briquette",
    "ref_bucket": "REF_02 = KIK bucket / ведро examples -> JSON field has_bucket",
    "ref_cone": "REF_03 = KIK cone / рожок examples -> JSON field has_cone",
    "ref_cups": "REF_04 = KIK cup / стаканчик examples -> JSON field has_cup",
    "ref_eskimo": "REF_05 = KIK eskimo / эскимо examples -> JSON field has_eskimo",
    "ref_lakomka": "REF_06 = KIK lakomka examples -> JSON field has_lakomka",
    "ref_sandwich": "REF_07 = KIK sandwich examples -> JSON field has_sandwich",
}


@dataclass(frozen=True)
class EvalTaskSpec:
    name: str
    system_prompt: str
    user_prompt: str
    schema_instruction: str
    required_fields: list[str]
    response_format_payload: dict[str, Any]
    validate_prediction: Callable[[dict[str, Any]], Any]
    score_fields: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, float]]
    mock_prediction: dict[str, Any]
    resolve_labels_path: Callable[[Path | None], tuple[Path, str]]
    load_cases: Callable[[Path, Path, int | None], list[EvalCase]]
    aggregate_results: Callable[[list[dict[str, Any]]], dict[str, Any]]
    write_outputs: Callable[[Path, list[dict[str, Any]], dict[str, Any], int, Path, list[str]], None]
    reference_label: Callable[[Path, int], str]
    image_prompt_contract: dict[str, str] | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KIK retail execution VLM image-to-JSON benchmark.")
    parser.add_argument("--task", choices=["kik", "kik_simple"], default="kik")
    parser.add_argument("--images", type=Path, default=Path("data/real_images"))
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--include-heavy", action="store_true")
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--references", type=Path, default=DEFAULT_KIK_REFERENCES)
    parser.add_argument("--reference-sheet", type=Path, default=None)
    parser.add_argument("--no-references", action="store_true")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "models.yaml")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Reuse successful rows from an existing run dir and run only missing/failed image-model pairs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _load_dotenv_if_available()
    args = parse_args(argv)
    task_spec = get_task_spec(args.task)
    output_root = args.output or Path(f"runs/{task_spec.name}_eval")
    provider_override = args.provider or os.getenv("VLM_EVAL_PROVIDER")
    timeout_seconds = float(os.getenv("VLM_EVAL_TIMEOUT_SECONDS", "90"))
    all_models = load_models(args.config)
    models = select_models(all_models, args.models, args.include_heavy, provider_override)
    heavy_models = [model for model in models if model.heavy]
    if heavy_models:
        print("WARNING: heavy quality-ceiling models selected: " + ", ".join(model.key for model in heavy_models))
        print("They are not production/self-host candidates and may be expensive.")

    labels_path, label_mode = task_spec.resolve_labels_path(args.labels)
    cases = task_spec.load_cases(args.images, labels_path, args.limit)
    if not cases:
        print("No images found. Check --images and optional --labels.", file=sys.stderr)
        return 2
    reference_paths = [] if args.no_references else resolve_reference_paths(args, task_spec)
    if task_spec.name == "kik_simple" and args.no_references:
        print("WARNING: kik_simple without a reference sheet is intended only for mock/smoke runs.", file=sys.stderr)
    if task_spec.name == "kik_simple" and not args.no_references and len(reference_paths) != 1:
        print(
            "kik_simple expects exactly one reference-sheet image. "
            "Use --reference-sheet data/reference_images_sku_sheet/kik_sku_reference.png",
            file=sys.stderr,
        )
        return 2

    run_dir = make_run_dir(output_root)
    print(f"Run dir: {run_dir}")
    print(f"Task: {task_spec.name}")
    print(f"Dataset cases: {len(cases)}")
    print(f"Labels: {labels_path} ({label_mode})")
    print("References: " + (", ".join(path.name for path in reference_paths) if reference_paths else "none"))
    print("Models: " + ", ".join(model.key for model in models))
    print("KIK fields: " + ", ".join(task_spec.required_fields))

    resumed_successes = load_successful_resume_rows(args.resume_from) if args.resume_from else []
    allowed_keys = {(case.image, model.key) for model in models for case in cases}
    resumed_by_key = {
        result_key(row): row
        for row in resumed_successes
        if row.get("task") == task_spec.name and result_key(row) in allowed_keys
    }

    dump_config_snapshot(
        run_dir / "config_snapshot.yaml",
        {
            "task": task_spec.name,
            "images": str(args.images),
            "labels": str(labels_path),
            "labels_mode": label_mode,
            "models": args.models,
            "limit": args.limit,
            "concurrency": args.concurrency,
            "output": str(output_root),
            "include_heavy": args.include_heavy,
            "provider": provider_override,
            "references": ",".join(str(path) for path in reference_paths),
            "reference_count": len(reference_paths),
            "reference_sheet": str(args.reference_sheet) if args.reference_sheet else None,
            "timeout_seconds": timeout_seconds,
            "resume_from": str(args.resume_from) if args.resume_from else None,
            "resumed_success_count": len(resumed_by_key),
            "api_max_attempts": os.getenv("VLM_EVAL_API_MAX_ATTEMPTS", "5"),
            "retry_base_seconds": os.getenv("VLM_EVAL_RETRY_BASE_SECONDS", "10"),
            "retry_max_seconds": os.getenv("VLM_EVAL_RETRY_MAX_SECONDS", "120"),
            "image_prompt_contract": task_spec.image_prompt_contract or "default TARGET_00/REF_* image map",
        },
        models,
    )

    tasks = [(case, model) for model in models for case in cases]
    tasks = [
        (case, model)
        for case, model in tasks
        if (case.image, model.key) not in resumed_by_key
    ]
    results: list[dict[str, Any]] = list(resumed_by_key.values())
    if resumed_by_key:
        print(f"Resumed successful rows: {len(resumed_by_key)}")
    print(f"Pending model calls: {len(tasks)}")
    progress_path = run_dir / "progress.jsonl"
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [
            executor.submit(run_one, case, model, timeout_seconds, task_spec, reference_paths)
            for case, model in tasks
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            append_jsonl(progress_path, result)
            status = "ok" if not result.get("error") else "error"
            print(f"[{index}/{len(tasks)}] {result['model_key']} {result['image']} {status}")

    aggregates = task_spec.aggregate_results(results)
    task_spec.write_outputs(
        run_dir,
        results,
        aggregates,
        dataset_size=len(cases),
        labels_path=labels_path,
        models_tested=[model.key for model in models],
    )

    print(f"Saved results: {run_dir / 'results.jsonl'}")
    print(f"Saved summary: {run_dir / 'summary.md'}")
    error_count = sum(1 for result in results if result.get("error"))
    if error_count:
        print(
            f"WARNING: run contains {error_count} API/JSON/schema error rows. "
            "Do not treat aggregate quality metrics as clean until errors are retried."
        )
        print(
            "Retry only failed/missing rows with: "
            f"python -m vlm_eval.run --images {args.images} --references {args.references} "
            f"--models {args.models or ','.join(model.key for model in models)} "
            f"--resume-from {run_dir} --output {output_root}"
        )
    return 0


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def find_reference_images(reference_dir: Path) -> list[Path]:
    if reference_dir.is_file() and reference_dir.suffix.lower() in IMAGE_EXTENSIONS:
        return [reference_dir]
    if not reference_dir.exists():
        return []
    paths = [
        path
        for path in sorted(reference_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=lambda path: (REFERENCE_ORDER.get(path.stem, 999), path.name))


def load_successful_resume_rows(run_dir: Path | None) -> list[dict[str, Any]]:
    if run_dir is None:
        return []
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"Cannot resume: {results_path} does not exist")
    rows: list[dict[str, Any]] = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("error") and row.get("schema_valid"):
            rows.append(row)
    return rows


def result_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["image"]), str(row["model_key"])


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def reference_label(path: Path) -> str:
    return REFERENCE_LABELS.get(path.stem, path.stem.replace("_", " "))


def simple_reference_label(path: Path, index: int) -> str:
    if index == 1:
        return "REF_SKU_SHEET = KIK SKU reference sheet with many visual examples"
    return f"REF_SKU_{index:02d} = KIK SKU reference sheet {path.stem.replace('_', ' ')}"


def kik_reference_label(path: Path, index: int) -> str:
    return reference_label(path)


def resolve_reference_paths(args: argparse.Namespace, task_spec: EvalTaskSpec) -> list[Path]:
    if task_spec.name == "kik_simple":
        explicit_source = args.reference_sheet
        if explicit_source is None and args.references != DEFAULT_KIK_REFERENCES:
            explicit_source = args.references
        source = explicit_source or DEFAULT_KIK_SIMPLE_REFERENCE
        return find_reference_images(source)
    return find_reference_images(args.references)


def get_task_spec(task: str = "kik") -> EvalTaskSpec:
    if task == "kik":
        return EvalTaskSpec(
            name="kik",
            system_prompt=KIK_SYSTEM_PROMPT,
            user_prompt=KIK_USER_PROMPT,
            schema_instruction=kik_json_schema_instruction(),
            required_fields=KIK_REQUIRED_FIELDS,
            response_format_payload=kik_response_format_json_schema(),
            validate_prediction=validate_kik_prediction,
            score_fields=score_kik_fields,
            mock_prediction=make_mock_kik_prediction(),
            resolve_labels_path=resolve_kik_labels_path,
            load_cases=load_kik_cases,
            aggregate_results=aggregate_kik_by_model,
            write_outputs=write_kik_outputs,
            reference_label=kik_reference_label,
            image_prompt_contract=None,
        )
    if task == "kik_simple":
        return EvalTaskSpec(
            name="kik_simple",
            system_prompt=KIK_SIMPLE_SYSTEM_PROMPT,
            user_prompt=KIK_SIMPLE_USER_PROMPT,
            schema_instruction=kik_simple_json_schema_instruction(),
            required_fields=KIK_SIMPLE_REQUIRED_FIELDS,
            response_format_payload=kik_simple_response_format_json_schema(),
            validate_prediction=validate_kik_simple_prediction,
            score_fields=score_kik_simple_fields,
            mock_prediction=make_mock_kik_simple_prediction(),
            resolve_labels_path=resolve_kik_simple_labels_path,
            load_cases=load_kik_simple_cases,
            aggregate_results=aggregate_kik_simple_by_model,
            write_outputs=write_kik_simple_outputs,
            reference_label=simple_reference_label,
            image_prompt_contract=kik_simple_image_prompt_contract(),
        )
    raise ValueError(f"Unsupported task: {task}")


def extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Top-level JSON must be an object")
    return parsed


def run_one(
    case: EvalCase,
    model: ModelConfig,
    timeout_seconds: float,
    task_spec: EvalTaskSpec | None = None,
    reference_paths: list[Path] | None = None,
) -> dict[str, Any]:
    task_spec = task_spec or get_task_spec()
    base = {
        "image": case.image,
        "task": task_spec.name,
        "model_key": model.key,
        "model": model.canonical_model,
        "role": model.role,
        "provider": model.provider,
        "provider_model": model.provider_model,
        "latency_ms": None,
        "json_parse_ok": False,
        "schema_valid": False,
        "retry_count": 0,
        "api_retry_count": 0,
        "token_usage": None,
        "raw_response": "",
        "parsed": None,
        "expected": case.expected,
        "field_scores": {},
        "error": None,
        "response_format_mode": None,
    }
    try:
        image_data_url = encode_image_data_url(case.image_path, model.image_max_side)
        reference_images = [
            (task_spec.reference_label(path, index), encode_image_data_url(path, model.image_max_side))
            for index, path in enumerate(reference_paths or [], start=1)
        ]
        provider = create_provider(model.provider)
    except Exception as exc:
        base["error"] = str(exc)
        return base

    last_error: str | None = None
    last_row: dict[str, Any] | None = None
    for response_format_mode in response_format_candidates(model):
        for retry_count in range(2):
            try:
                response = provider.complete(
                    model,
                    image_data_url,
                    response_format_mode,
                    timeout_seconds,
                    system_prompt=task_spec.system_prompt,
                    user_prompt=task_spec.user_prompt,
                    schema_instruction=task_spec.schema_instruction,
                    response_format_payload=task_spec.response_format_payload,
                    mock_prediction=task_spec.mock_prediction,
                    reference_images=reference_images,
                    image_prompt_contract=task_spec.image_prompt_contract,
                )
                row = dict(base)
                row.update(
                    {
                        "latency_ms": response.latency_ms,
                        "retry_count": retry_count,
                        "api_retry_count": response.api_retry_count,
                        "token_usage": response.token_usage,
                        "raw_response": response.raw_response,
                        "response_format_mode": response.response_format_mode,
                    }
                )
                try:
                    parsed = extract_json_object(response.raw_response)
                except Exception as exc:
                    last_error = f"JSON parse error: {exc}"
                    row["error"] = last_error
                    last_row = row
                    if retry_count == 0:
                        continue
                    break
                validation = task_spec.validate_prediction(parsed)
                row.update(
                    {
                        "json_parse_ok": True,
                        "schema_valid": validation.ok,
                        "parsed": parsed,
                        "field_scores": task_spec.score_fields(case.expected, parsed),
                        "error": None if validation.ok else "; ".join(validation.errors),
                    }
                )
                last_row = row
                if validation.ok:
                    return row
                if retry_count == 1:
                    break
                last_error = row["error"]
            except ProviderError as exc:
                last_error = str(exc)
                if model.provider == "openrouter" and exc.status_code == 400 and response_format_mode != "none":
                    break
                row = dict(base)
                row.update({"error": str(exc), "response_format_mode": response_format_mode})
                return row
            except Exception as exc:
                last_error = str(exc)
                if retry_count == 1:
                    row = dict(base)
                    row.update({"error": str(exc), "retry_count": retry_count, "response_format_mode": response_format_mode})
                    return row
    row = dict(base)
    row["error"] = last_error or "unknown error"
    return last_row or row


if __name__ == "__main__":
    raise SystemExit(main())
