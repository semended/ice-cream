# Ice Cream VLM MVP

KIK-only VLM benchmark for auditing retail ice-cream equipment photos.

## Demo · фото-контроль точки

Финальная product demo: **[`kik-product-demo-final.html`](./kik-product-demo-final.html)** в корне проекта. Один сценарий: фото торговой точки (`photo_006`) → разбор представленности КИК → задачи для ТП. Standalone HTML, без бэкенда и сборки.

Открыть локально:

```bash
open kik-product-demo-final.html
```

Опубликовать на GitHub Pages: `Settings → Pages → Source: Deploy from a branch → Branch: main / (root) → Save`. После активации ссылка: `https://semended.github.io/ice-cream/kik-product-demo-final.html`.

Используемое фото: `assets/demo/photo_006.jpg`.

There are two task branches:

- `kik` default: old flow with SKU-family classification fields and 7 separate reference images;
- `kik_simple`: simplified SKU lookup flow with one SKU reference sheet, no SKU-family classification, and the same GT core fields.

The project compares vision-language models on business fields for «Коровка из Кореновки» / КИК:

- KIK presence;
- visible KIK SKU count;
- KIK share percent;
- retail/ice-cream equipment check and full-crop flag;
- SKU families;
- POSM and monobrand block;
- mixed competitor placement;
- non-ice-cream products, empty sections, foreign labels;
- final status: normal / attention / critical.

## Current Structure

```text
data/
  real_images/                 target JPG photos
  reference_images_slides/     selected KIK visual references
  reference_images_sku_sheet/   optional single reference sheet for --task kik_simple
  reference_candidates/        backup source crops/candidates for references
  ground_truth/
    kik_report_ground_truth.csv
    kik_report_ground_truth_template.csv
    manual_ground_truth.jsonl
  raw/                         source PDFs

vlm_eval/
  run.py                       KIK benchmark runner
  models.yaml                  model config
  tasks/kik/                   KIK prompt, schema, scoring, reporting

src/
  generate_kik_executive_report.py

runs/
  kik_eval_7x10_merged/        kept benchmark run and HTML report
```

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put API keys in `.env`:

```env
OPENROUTER_API_KEY=...
DEEPINFRA_API_KEY=...
```

## Smoke Test

No API calls:

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

## Run Benchmark

Production candidates:

```bash
python -m vlm_eval.run \
  --images data/real_images \
  --references data/reference_images_slides \
  --models qwen3_vl_30b,qwen25_vl_72b,gemma4_31b,glm_46v,mistral_small_4 \
  --concurrency 3 \
  --output runs/kik_eval
```

Full 7-model benchmark with heavy quality ceilings:

```bash
python -m vlm_eval.run \
  --images data/real_images \
  --references data/reference_images_slides \
  --models qwen3_vl_235b,mistral_large_3,qwen3_vl_30b,qwen25_vl_72b,gemma4_31b,glm_46v,mistral_small_4 \
  --include-heavy \
  --concurrency 3 \
  --output runs/kik_eval
```

Retry or continue a partially failed run without re-running successful image/model pairs:

```bash
python -m vlm_eval.run \
  --images data/real_images \
  --references data/reference_images_slides \
  --models gemma4_31b \
  --concurrency 1 \
  --resume-from runs/kik_eval/<failed_timestamp> \
  --output runs/kik_eval
```

For rate-limited hosted providers, tune retry behavior with:

```bash
export VLM_EVAL_API_MAX_ATTEMPTS=5
export VLM_EVAL_RETRY_BASE_SECONDS=10
export VLM_EVAL_RETRY_MAX_SECONDS=120
```

Direct Google AI Studio / Gemini API path for Gemma 4 31B:

```bash
export GEMINI_API_KEY=...
python -m vlm_eval.run \
  --images data/real_images \
  --references data/reference_images_slides \
  --models gemma4_31b \
  --provider google_aistudio \
  --concurrency 1 \
  --output runs/kik_eval_gemma4_31b_google_aistudio
```

Manual AI Studio UI responses can be scored with:

```bash
python src/score_manual_aistudio_predictions.py
```

Simplified SKU lookup benchmark with one reference sheet:

```bash
python -m vlm_eval.run \
  --task kik_simple \
  --images data/real_images \
  --reference-sheet data/reference_images_sku_sheet/kik_sku_reference.png \
  --models qwen3_vl_30b,qwen25_vl_72b,gemma4_31b,glm_46v,mistral_small_4 \
  --concurrency 3 \
  --output runs/kik_simple_eval
```

The runner loads only `.jpg` / `.jpeg` target images.

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

Generate the readable executive HTML report:

```bash
python src/generate_kik_executive_report.py \
  --run-dir runs/kik_eval/<timestamp> \
  --images-dir data/real_images
```

The current kept report is:

```text
runs/kik_eval_7x10_merged/20260510_142125/kik_executive_model_report.html
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Codex App

Project-scoped Codex App profiles live in `.codex/config.toml`, with usage notes in `.codex/README.md`.

Default profile: `power`.

Trust the nested repo root first if Codex reports that the profile is not found.

```bash
codex --cd /Users/semengolodnuk/Documents/ice_cream/ice-cream-vlm-mvp --profile power
codex --cd /Users/semengolodnuk/Documents/ice_cream/ice-cream-vlm-mvp --profile daily
```

## Notes

`data/reference_images_slides/` contains the active reference set for `--task kik`. The runner sends those references in the canonical REF_01..REF_07 order unless `--no-references` is passed.

`--task kik_simple` uses one reference-sheet image via `--reference-sheet`; by default it looks for `data/reference_images_sku_sheet/kik_sku_reference.png`. This branch removes SKU-family fields such as `has_cup`, `has_eskimo`, `has_bucket`, and keeps unique SKU count, KIK share, execution checks, and status.

The intended canonical KIK prompt/image-map contract is recorded in `docs/kik-canonical-prompt-contract.md`. The active runtime currently uses a simplified flat schema in `vlm_eval/tasks/kik/schema.py`; it keeps only business-visible KIK fields and maps brick/log into `has_poleno_or_briquette`.
