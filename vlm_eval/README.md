# KIK VLM Eval

KIK-only benchmark runner for retail execution photos of ice-cream trade equipment.

The default `kik` task is the old flow with SKU-family classification fields. The `kik_simple` task is a separate simplified flow with one SKU reference sheet and no SKU-family classification.

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
data/reference_images_sku_sheet/   optional one-image SKU sheet for --task kik_simple
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

Simplified SKU lookup smoke test:

```bash
python -m vlm_eval.run \
  --task kik_simple \
  --models mock \
  --limit 2 \
  --output runs/kik_simple_eval \
  --no-references
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

Simplified SKU lookup run with one SKU sheet:

```bash
OPENROUTER_API_KEY=... python -m vlm_eval.run \
  --task kik_simple \
  --images data/real_images \
  --reference-sheet data/reference_images_sku_sheet/kik_sku_reference.png \
  --models qwen3_vl_30b,qwen25_vl_72b,gemma4_31b,glm_46v,mistral_small_4 \
  --concurrency 3 \
  --output runs/kik_simple_eval
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

Resume a partially failed run without re-running successful rows:

```bash
OPENROUTER_API_KEY=... python -m vlm_eval.run \
  --images data/real_images \
  --references data/reference_images_slides \
  --models gemma4_31b \
  --concurrency 1 \
  --resume-from runs/kik_eval/<failed_timestamp> \
  --output runs/kik_eval
```

## Config

Models are configured in `vlm_eval/models.yaml`.

Environment:

```bash
export OPENROUTER_API_KEY=...
export GEMINI_API_KEY=...
export DEEPINFRA_API_KEY=...
export LOCAL_VLLM_BASE_URL=http://127.0.0.1:8000/v1
export LOCAL_VLLM_API_KEY=local-token
export VLM_EVAL_PROVIDER=openrouter
export VLM_EVAL_TIMEOUT_SECONDS=90
export VLM_EVAL_API_MAX_ATTEMPTS=5
export VLM_EVAL_RETRY_BASE_SECONDS=10
export VLM_EVAL_RETRY_MAX_SECONDS=120
```

Run Gemma 4 31B through Google AI Studio / Gemini API instead of OpenRouter:

```bash
GEMINI_API_KEY=... python -m vlm_eval.run \
  --images data/real_images \
  --references data/reference_images_slides \
  --models gemma4_31b \
  --provider google_aistudio \
  --concurrency 1 \
  --output runs/kik_eval_gemma4_31b_google_aistudio
```

If responses are copied from AI Studio UI manually, score them with:

```bash
python src/score_manual_aistudio_predictions.py
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
