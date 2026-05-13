from __future__ import annotations

SYSTEM_PROMPT = """You analyze photos of ice cream retail equipment for the brand “Korovka iz Korenovki” / KIK.

Return only valid JSON according to the required schema.
Do not add markdown, explanations, reasoning, comments, or any text outside the JSON.

KIK = “Korovka iz Korenovki” / “Коровка из Кореновки”.

This is the simplified SKU lookup task:
- do not classify products into SKU families or package types;
- use the reference sheet only to recognize KIK SKU examples and packaging identity;
- count visible unique KIK SKUs in the target photo;
- estimate KIK share in the upper visible part of the freezer/display;
- evaluate the reduced execution and status fields from the schema.

You may receive several images in one request:
- TARGET_00 is the only real retail equipment photo to analyze.
- REF_SKU_SHEET is a visual reference sheet with many KIK SKU examples.
- Use REF_SKU_SHEET only to recognize KIK products and decide whether visible packages are the same or different unique SKUs.
- Do not count products from REF_SKU_SHEET as target evidence.
- Do not use REF_SKU_SHEET to decide share, monobrand block, competitor mixing, non-ice-cream products, or final status.

Unique SKU counting:
- count unique visible KIK SKUs, not package facings;
- the same SKU repeated many times counts as 1;
- different visible packaging, flavor, product name, or format counts as different SKUs;
- if two packages may be the same SKU and there is not enough visual evidence, count conservatively as one SKU.

KIK share percent:
- estimate the percentage of KIK products in the upper visible part of the freezer/display in TARGET_00;
- numerator = visible area/facings occupied by KIK in that upper visible part;
- denominator = the whole upper visible freezer/display area, not only branded products;
- return an integer from 0 to 100.

If KIK is not visible, set:
`kik_present=false`,
`kik_sku_count=0`,
`kik_share_percent=0`."""

USER_PROMPT = """Analyze TARGET_00 and return JSON according to the simplified KIK SKU lookup schema.

IMAGE MAP:
- TARGET_00 = target retail equipment photo to analyze.
- REF_SKU_SHEET = one reference sheet containing many KIK SKU examples.

Important:
- Analyze only TARGET_00.
- Use REF_SKU_SHEET only as a visual SKU dictionary for KIK packaging examples.
- Do not count any product from REF_SKU_SHEET.
- Do not classify SKU families such as cup, eskimo, cone, bucket, sandwich, or log.
- Count only unique visible KIK SKUs in TARGET_00.
- Estimate KIK share only from the upper visible part of the freezer/display in TARGET_00.

Pay special attention to:

1. Whether TARGET_00 is retail equipment or not.
2. Whether TARGET_00 contains ice cream or not.
3. Whether “Korovka iz Korenovki” / KIK products are present in TARGET_00.
4. The number of unique visible KIK SKUs in TARGET_00.
5. The approximate KIK share percent in the upper visible part of the freezer/display in TARGET_00.
6. Whether there is a monobrand KIK block in TARGET_00.
7. Whether KIK is mixed with competitor products in TARGET_00.
8. Whether there are foreign non-ice-cream products in TARGET_00.
9. Final status:
    - 0 = NORMAL;
    - 1 = ATTENTION;
    - 2 = CRITICAL.

Return only JSON."""


def json_schema_instruction() -> str:
    return """Simplified KIK SKU lookup schema fields:
- is_trade_equipment_photo: boolean
- is_ice_cream_equipment: boolean
- kik_present: boolean
- kik_sku_count: integer 0..100, unique visible KIK SKUs only
- kik_share_percent: integer 0..100, KIK share in the upper visible part of the freezer/display
- has_monobrand_block, has_non_icecream_products, is_kik_mixed_with_competitors: boolean or null
- status_score: integer 0..2, where 0=НОРМА, 1=ВНИМАНИЕ, 2=КРИТИЧНО

All fields are required. No extra fields. No SKU family classification fields. No markdown. No text outside JSON."""


def image_prompt_contract() -> dict[str, str]:
    return {
        "target_id": "TARGET_00",
        "target_position": "after_references",
        "target_map_line": "TARGET_00 = target retail equipment photo to analyze.",
        "role_rules": (
            "Use TARGET_00 as the only real retail equipment photo to analyze. "
            "Use REF_SKU_SHEET only as a visual reference sheet for KIK SKU examples. "
            "Never count REF_SKU_SHEET products in kik_sku_count, kik_share_percent, "
            "status_score, non-ice-cream products, or competitor mixing. "
            "Do not classify SKU families in this task."
        ),
        "target_intro": (
            "TARGET_00 image follows. This is the only real retail equipment photo "
            "to analyze and score."
        ),
        "reference_intro_template": (
            "{reference_id} reference sheet follows. {reference_map_line}. "
            "Use {reference_id} only to recognize KIK packaging examples and to decide "
            "whether visible TARGET_00 packages are the same or different unique SKUs. "
            "Do not classify SKU families and do not score this reference image."
        ),
        "final_task_intro": (
            "FINAL TASK: analyze TARGET_00 only. Compare visible TARGET_00 products with "
            "REF_SKU_SHEET only for KIK identity and unique SKU counting. Return only one "
            "complete JSON object. Do not explain, reason step by step, use markdown, or "
            "output text before or after the JSON. The first character must be { and the "
            "last character must be }."
        ),
    }
