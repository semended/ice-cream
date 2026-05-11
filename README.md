# Ice Cream VLM MVP

KIK-only VLM benchmark for auditing retail ice-cream equipment photos.

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

`data/reference_images_slides/` contains the active reference set. The runner sends references in the canonical REF_01..REF_07 order unless `--no-references` is passed.

The intended canonical KIK prompt/image-map contract is recorded in `docs/kik-canonical-prompt-contract.md`. The active runtime currently uses a simplified flat schema in `vlm_eval/tasks/kik/schema.py`; it keeps only business-visible KIK fields and maps brick/log into `has_poleno_or_briquette`.
