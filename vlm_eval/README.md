# KIK VLM Eval

KIK-only benchmark runner for retail execution photos of ice-cream trade equipment.

The model receives one JPG target photo from `data/real_images` and must return strict JSON for KIK business fields:

- KIK presence;
- visible KIK SKU count;
- KIK share percent;
- retail/ice-cream equipment check and full-crop flag;
- SKU families;
- POSM and monobrand block;
- mixed competitor placement;
- non-ice-cream products, empty sections, foreign labels;
- final action status.

## Data

```text
data/real_images/                 target JPG images
data/ground_truth/manual_ground_truth.jsonl
data/reference_images_slides/     selected visual references used by default
```

Only `.jpg` / `.jpeg` images are loaded by the runner.

## Smoke Test

No API call:

```bash
python -m vlm_eval.run \
  --models mock \
  --limit 2 \
  --output runs/kik_eval
```

## Real Run

```bash
OPENROUTER_API_KEY=... python -m vlm_eval.run \
  --images data/real_images \
  --references data/reference_images_slides \
  --models qwen3_vl_30b,qwen25_vl_72b,gemma4_31b,glm_46v,mistral_small_4 \
  --concurrency 3 \
  --output runs/kik_eval
```

Full benchmark with heavy quality-ceiling models:

```bash
OPENROUTER_API_KEY=... python -m vlm_eval.run \
  --images data/real_images \
  --references data/reference_images_slides \
  --models qwen3_vl_235b,mistral_large_3,qwen3_vl_30b,qwen25_vl_72b,gemma4_31b,glm_46v,mistral_small_4 \
  --include-heavy \
  --concurrency 3 \
  --output runs/kik_eval
```

## Config

Models are configured in `vlm_eval/models.yaml`.

Environment:

```bash
export OPENROUTER_API_KEY=...
export DEEPINFRA_API_KEY=...
export LOCAL_VLLM_BASE_URL=http://127.0.0.1:8000/v1
export LOCAL_VLLM_API_KEY=local-token
export VLM_EVAL_PROVIDER=openrouter
export VLM_EVAL_TIMEOUT_SECONDS=90
```

## Outputs

Each run creates:

```text
runs/kik_eval/<timestamp>/
  results.jsonl
  errors.jsonl
  summary.csv
  summary.md
  boolean_metrics_by_model.csv
  numeric_metrics_by_model.csv
  business_key_metrics_by_model.csv
  field_coverage_by_model.csv
  worst_cases_by_model.csv
  confusion_status_score.csv
  config_snapshot.yaml
```

Use `src/generate_kik_executive_report.py` to generate the readable HTML report from a run directory.
