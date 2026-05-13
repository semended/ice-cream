# Current Gemma 4 31B Run Metrics

Run:
`runs/kik_eval_gemma4_31b_aistudio_batch_fields_removed_final/20260511_204132`

## Key results

| Metric | Value |
|---|---:|
| JSON valid rate | 100% |
| Schema valid rate | 100% |
| KIK present F1 | 1.0 |
| KIK share MAE | 16.5 pp |
| SKU count MAE | 4.1 |

## Interpretation

Gemma already returns structured business fields and reliably detects KIK presence.

However:
- exact SKU count is unstable;
- exact KIK share is unstable on difficult photos;
- outlet status is often underestimated as normal.

## Product conclusion

VLM is useful for initial business signal extraction, but industrial MVP should use a hybrid pipeline:

1. Segmentation model for share, zones, empty sections and mixing.
2. VLM for complex visual/business factors.
3. Business rules for final status and recommendation.
