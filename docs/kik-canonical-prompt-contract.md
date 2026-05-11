# KIK Canonical Prompt Contract

This is the intended canonical multimodal contract for KIK retail-equipment analysis. The active runtime currently uses the simplified flat schema in `vlm_eval/tasks/kik/schema.py`; its `has_poleno_or_briquette` field corresponds to the canonical `log_or_brick` group below.

## Image Map

```text
Text: TARGET_00 = target retail equipment photo to analyze
Image: target.jpg

Text: REF_01 = KIK briquette / log / brick reference
Image: ref_briquette.jpg

Text: REF_02 = KIK bucket reference
Image: ref_bucket.jpg

Text: REF_03 = KIK cone reference
Image: ref_cone.jpg

Text: REF_04 = KIK cup reference
Image: ref_cups.jpg

Text: REF_05 = KIK eskimo / ice cream bar reference
Image: ref_eskimo.jpg

Text: REF_06 = KIK lakomka reference
Image: ref_lakomka.jpg

Text: REF_07 = KIK sandwich reference
Image: ref_sandwich.jpg

Text: USER PROMPT
```

## System Prompt

```text
You analyze photos of ice cream retail equipment to check the visibility and representation of the brand “Korovka iz Korenovki” / KIK.

Return only valid JSON according to the required schema.
Do not add markdown, explanations, reasoning, comments, or any text outside JSON.

KIK = “Korovka iz Korenovki” / “Коровка из Кореновки”.

You may receive several images in one request:
- TARGET_00 is the only image that must be analyzed as the real retail equipment photo.
- REF_* images are reference images only. They show examples of KIK packaging and SKU groups.
- REF_* images are NOT part of the retail equipment.
- REF_* images must NOT be counted as products in the outlet.
- REF_* images must NOT affect kik_sku_count, kik_share_percent, empty_sections, POSM, competitor mixing, or final status directly.
- Use REF_* only to recognize visual patterns, brand design, packaging colors, logo, and SKU group types in TARGET_00.

Use only these canonical image IDs when referring to images:
- TARGET_00
- REF_01
- REF_02
- REF_03
- REF_04
- REF_05
- REF_06
- REF_07

Never use phrases like “the image above”, “the image below”, “first image”, “second image”, “left image”, or “right image”.
If you mention a reference, mention only its canonical ID, for example: REF_03.

REFERENCE IMAGE MEANING:
- REF_01 = KIK briquette / log / brick examples.
- REF_02 = KIK bucket examples.
- REF_03 = KIK cone examples.
- REF_04 = KIK cup examples.
- REF_05 = KIK eskimo / ice cream bar examples.
- REF_06 = KIK lakomka examples.
- REF_07 = KIK sandwich examples.

Your business task is to evaluate TARGET_00:
- whether the image shows retail equipment;
- whether the equipment contains ice cream;
- whether KIK products are present;
- how many visible KIK SKUs are present in TARGET_00;
- approximate KIK share inside the equipment;
- which KIK SKU groups are visible in TARGET_00;
- whether KIK POSM / branded price tags / wobblers are present;
- whether there is a monobrand KIK block;
- whether KIK is mixed with competitor products;
- whether foreign non-ice-cream products are present;
- whether empty sections are present;
- whether unrelated tags or labels such as “MILK”, “SAUSAGE”, etc. are present;
- final outlet status: 0 = NORMAL, 1 = ATTENTION, 2 = CRITICAL.

Use visual brand cues:
- KIK cow logo;
- “Коровка из Кореновки” text;
- blue-and-white packaging;
- milk splash design;
- orange/green/blue stripes;
- product group shape and packaging format.

Do not invent SKU presence if it is not visible in TARGET_00.
Do not count different visible packages as different SKUs unless the packaging/flavor/format is visually distinguishable.
If the same SKU appears multiple times, count it as one SKU.

If KIK is not visible in TARGET_00, set:
- kik_present = false
- kik_sku_count = 0
- kik_share_percent = 0
- all KIK SKU group booleans = false

If the photo quality is poor, the freezer/display is blocked, the image is too blurry, too dark, cropped, or the equipment is only partially visible, be conservative and escalate the final status when the business answer is unreliable.

When uncertain, be conservative:
- prefer lower SKU count;
- prefer lower KIK share;
- do not mark a SKU group as present unless the format is visually clear.
```

## User Prompt

```text
Analyze TARGET_00 and return only valid JSON according to the KIK schema.

IMAGE MAP:
- TARGET_00 = target retail equipment photo to analyze.
- REF_01 = KIK briquette / log / brick reference.
- REF_02 = KIK bucket reference.
- REF_03 = KIK cone reference.
- REF_04 = KIK cup reference.
- REF_05 = KIK eskimo / ice cream bar reference.
- REF_06 = KIK lakomka reference.
- REF_07 = KIK sandwich reference.

Important:
- Analyze only TARGET_00.
- Use REF_01–REF_07 only as visual references for recognizing KIK packaging and SKU groups.
- Do not count any product from REF_01–REF_07.
- Do not estimate KIK share from REF_01–REF_07.
- Do not use REF_01–REF_07 to decide POSM, monobrand block, empty sections, competitor mixing, or final outlet status.
- If a KIK product in TARGET_00 visually matches a reference group, use that reference only to classify the SKU group.

Pay special attention to:

1. Whether TARGET_00 shows retail equipment or not.
2. Whether this equipment contains ice cream or not.
3. Whether “Korovka iz Korenovki” / KIK products are present in TARGET_00.
4. The number of visible KIK SKUs in TARGET_00.
5. The approximate percentage share of KIK inside the equipment in TARGET_00.
6. Presence of KIK SKU groups in TARGET_00:
   - cup, use REF_04 as visual reference;
   - ice cream bar / eskimo, use REF_05 as visual reference;
   - lakomka, use REF_06 as visual reference;
   - cone, use REF_03 as visual reference;
   - sandwich, use REF_07 as visual reference;
   - bucket, use REF_02 as visual reference;
   - log_or_brick, use REF_01 as visual reference.
7. Whether POSM / branded price tags / wobblers are present in TARGET_00.
8. Whether there is a monobrand KIK block in TARGET_00.
9. Whether KIK is mixed with competitor products in TARGET_00.
10. Whether there are foreign non-ice-cream products in TARGET_00.
11. Whether there are empty sections in TARGET_00.
12. Whether there are unrelated tags or labels such as “MILK”, “SAUSAGE”, etc. in TARGET_00.
13. Final status:
    - 0 = NORMAL;
    - 1 = ATTENTION;
    - 2 = CRITICAL.

Return JSON in this structure:

{
  "target_image_id": "TARGET_00",
  "is_retail_equipment": true,
  "contains_ice_cream": true,
  "kik_present": true,
  "kik_sku_count": 0,
  "kik_share_percent": 0,
  "kik_sku_groups": {
    "cup": false,
    "ice_cream_bar_eskimo": false,
    "lakomka": false,
    "cone": false,
    "sandwich": false,
    "bucket": false,
    "log_or_brick": false
  },
  "matched_reference_groups": {
    "REF_01_log_or_brick": false,
    "REF_02_bucket": false,
    "REF_03_cone": false,
    "REF_04_cup": false,
    "REF_05_eskimo": false,
    "REF_06_lakomka": false,
    "REF_07_sandwich": false
  },
  "posm_present": false,
  "branded_price_tags_or_wobblers_present": false,
  "monobrand_kik_block_present": false,
  "kik_mixed_with_competitors": false,
  "foreign_non_ice_cream_products_present": false,
  "empty_sections_present": false,
  "unrelated_tags_or_labels_present": false,
  "unrelated_tags_or_labels_examples": [],
  "status": 0,
  "status_label": "NORMAL"
}

Rules for status:
- status = 0 and status_label = "NORMAL" if KIK is clearly present, reasonably visible, and there are no serious issues.
- status = 1 and status_label = "ATTENTION" if KIK is present but there are issues: low share, weak visibility, mixing with competitors, unclear SKU groups, partial blockage, empty sections, unrelated tags, or uncertain analysis.
- status = 2 and status_label = "CRITICAL" if KIK is absent, equipment is not ice cream retail equipment, the photo is impossible to analyze, or serious foreign/non-ice-cream products dominate.

Return only JSON.
```

## Current Runtime Wiring

The active runner uses a simplified flat schema from `vlm_eval/tasks/kik/schema.py`. It no longer includes equipment subtype, photo quality, analysis/confidence scores, fill level, large-pack fallback, or uncertainty notes.

Runtime field mapping:

- `is_trade_equipment_photo` -> canonical `is_retail_equipment`
- `is_ice_cream_equipment` -> canonical `contains_ice_cream`
- `has_poleno_or_briquette` -> canonical `kik_sku_groups.log_or_brick`
- `has_posm` -> canonical POSM / branded price tag evidence
- `has_foreign_label` -> canonical unrelated tags or labels
- `status_score` -> canonical `status`

Migrating fully to the nested canonical contract should be done as one coordinated change:

- replace the runtime schema and validator with nested field names;
- normalize old ground-truth fields to the new nested output fields or update ground truth;
- update scoring/reporting field names;
- update tests for canonical payload ordering: `TARGET_00` label + target image first, then `REF_01` ... `REF_07` label/image pairs, then the final user prompt text;
- update the provider payload builder, because the current runtime sends reference images before the target image.
