# Presentation Design Package

This archive contains source materials for designing a business demo deck for the Ice Cream VLM MVP project.

## Contents

- `photos/` — 10 real freezer photos used in the experiment.
- `input_data.md` — project context, hero case, problem case, positioning and metrics.
- `batch_table.csv` — short batch-analysis table for 10 photos.
- `json_examples/` — selected Gemma and ground-truth JSON examples.
- `metrics/run_metrics.md` — current run metrics and product interpretation.

## Main design direction

The presentation should look like a product demo deck, not a technical research report.

Core visual logic:

`field photo → AI analysis → business status → field recommendation`

## Mandatory design cases

### Hero case

Use `photos/photo_006.jpg`.

Purpose:
Show that the model can turn a real photo into a useful business signal.

### Problem case

Use `photos/photo_004.jpg`.

Purpose:
Show that VLM alone is not enough and that the MVP needs a segmentation pipeline.

## Target MVP pipeline

`photo → labeling → segmentation → VLM → business rules → dashboard/report`

## Do not overemphasize

- Full JSON
- Raw prompts
- Model zoo
- OCR
- Long technical explanation
- “Replacing supervisors”

Use this positioning instead:

“Automates primary photo control and highlights deviations for the field team.”
