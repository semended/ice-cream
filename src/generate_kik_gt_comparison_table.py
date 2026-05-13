from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_eval.tasks.kik.scoring import (
    BUSINESS_WEIGHTS,
    business_scores,
    score_kik_value,
)


FIELD_GROUPS = [
    (
        "Core",
        [
            "kik_present",
            "kik_sku_count",
            "kik_share_percent",
            "status_score",
        ],
    ),
    (
        "SKU",
        [
            "has_cup",
            "has_eskimo",
            "has_lakomka",
            "has_cone",
            "has_sandwich",
            "has_bucket",
            "has_poleno_or_briquette",
        ],
    ),
    (
        "Execution",
        [
            "has_posm",
            "has_monobrand_block",
            "has_non_icecream_products",
            "is_kik_mixed_with_competitors",
        ],
    ),
    (
        "Equipment",
        [
            "is_trade_equipment_photo",
            "is_ice_cream_equipment",
        ],
    ),
]

FIELD_LABELS = {
    "is_trade_equipment_photo": "trade eq",
    "is_ice_cream_equipment": "ice cream eq",
    "photo_crop_is_full": "crop full",
    "kik_present": "KIK",
    "kik_sku_count": "SKU count",
    "kik_share_percent": "share %",
    "has_cup": "cup",
    "has_eskimo": "eskimo",
    "has_lakomka": "lakomka",
    "has_cone": "cone",
    "has_sandwich": "sandwich",
    "has_bucket": "bucket",
    "has_poleno_or_briquette": "brick/log",
    "has_posm": "POSM",
    "has_monobrand_block": "monobrand",
    "has_foreign_label": "foreign label",
    "has_non_icecream_products": "non-icecream",
    "has_empty_sections": "empty sections",
    "is_kik_mixed_with_competitors": "mixed",
    "status_score": "status",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GT vs prediction comparison HTML table.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=Path("data/real_images"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = read_jsonl(args.run_dir / "results.jsonl")
    output = args.output or args.run_dir / "kik_gt_comparison_table.html"
    output.write_text(render_html(results, args.images_dir, output), encoding="utf-8")
    print(output)
    return 0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def render_html(results: list[dict[str, Any]], images_dir: Path, output: Path) -> str:
    rows = sorted(results, key=lambda row: row["image"])
    summary = summarize(rows)
    fields = [field for _group, group_fields in FIELD_GROUPS for field in group_fields]
    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KIK GT vs Gemma 4 31B</title>
<style>
  :root {{
    --bg: #f5f7fb;
    --text: #16202a;
    --muted: #697586;
    --line: #d9e0e8;
    --good: #dff7e5;
    --good-border: #67b77a;
    --bad: #ffe3df;
    --bad-border: #d86c61;
    --partial: #fff2c2;
    --partial-border: #c69b23;
    --card: #ffffff;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font: 13px/1.35 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  header {{
    padding: 22px 28px 16px;
    border-bottom: 1px solid var(--line);
    background: var(--card);
    position: sticky;
    top: 0;
    z-index: 4;
  }}
  h1 {{ margin: 0 0 12px; font-size: 22px; }}
  .metrics {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .metric {{
    min-width: 138px;
    padding: 9px 11px;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: #fbfcfe;
  }}
  .metric span {{ display: block; color: var(--muted); font-size: 11px; text-transform: uppercase; }}
  .metric b {{ display: block; margin-top: 3px; font-size: 18px; }}
  main {{ padding: 18px 28px 32px; }}
  .note {{ margin: 0 0 14px; color: var(--muted); }}
  .table-wrap {{
    overflow: auto;
    border: 1px solid var(--line);
    border-radius: 10px;
    background: var(--card);
  }}
  table {{
    border-collapse: separate;
    border-spacing: 0;
    min-width: 2300px;
    width: 100%;
  }}
  th, td {{
    border-right: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
    padding: 8px;
    vertical-align: top;
    background: var(--card);
  }}
  th {{
    position: sticky;
    top: 108px;
    z-index: 3;
    background: #eef3f8;
    font-size: 12px;
    text-align: left;
    white-space: nowrap;
  }}
  th.group {{ text-align: center; background: #e4ebf3; }}
  .sticky {{
    position: sticky;
    left: 0;
    z-index: 2;
    min-width: 180px;
    max-width: 180px;
    background: #fff;
  }}
  th.sticky {{ z-index: 5; background: #e4ebf3; }}
  .score-col {{
    position: sticky;
    left: 180px;
    z-index: 2;
    min-width: 92px;
    background: #fff;
  }}
  th.score-col {{ z-index: 5; background: #e4ebf3; }}
  .photo {{ display: grid; grid-template-columns: 74px 1fr; gap: 8px; align-items: center; }}
  .photo img {{ width: 74px; height: 74px; object-fit: cover; border-radius: 6px; border: 1px solid var(--line); }}
  .photo b {{ display: block; margin-bottom: 4px; }}
  .photo span {{ color: var(--muted); font-size: 11px; }}
  .cell {{
    min-width: 96px;
    border-left: 4px solid transparent;
    border-radius: 6px;
    padding: 6px;
  }}
  .ok {{ background: var(--good); border-left-color: var(--good-border); }}
  .bad {{ background: var(--bad); border-left-color: var(--bad-border); }}
  .partial {{ background: var(--partial); border-left-color: var(--partial-border); }}
  .na {{ background: #f2f4f7; color: var(--muted); }}
  .cell small {{ display: block; color: var(--muted); font-size: 10px; }}
  .cell strong {{ display: block; font-size: 12px; }}
  .field-summary {{ margin-top: 18px; }}
  .field-summary table {{ min-width: 720px; }}
  .field-summary th {{ position: static; }}
</style>
</head>
<body>
<header>
  <h1>KIK GT vs Gemma 4 31B AI Studio</h1>
  <div class="metrics">
    {metric("Business", fmt_pct(summary["business_score"]))}
    {metric("Core", fmt_pct(summary["core_score"]))}
    {metric("SKU family", fmt_pct(summary["sku_score"]))}
    {metric("Execution", fmt_pct(summary["execution_score"]))}
    {metric("Equipment", fmt_pct(summary["equipment_score"]))}
    {metric("Status", fmt_pct(summary["status_score"]))}
    {metric("Exact fields", f'{summary["matched_fields"]}/{summary["scored_fields"]}')}
  </div>
</header>
<main>
  <div class="table-wrap">
    <table>
      {render_header()}
      <tbody>
        {"".join(render_result_row(row, fields, images_dir) for row in rows)}
      </tbody>
    </table>
  </div>
  <section class="field-summary">
    <h2>Field Accuracy</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Field</th><th>Accuracy / avg score</th><th>Matched</th><th>Scored</th></tr></thead>
        <tbody>{"".join(render_field_summary_row(field, rows) for field in fields)}</tbody>
      </table>
    </div>
  </section>
</main>
</body>
</html>
"""


def render_header() -> str:
    group_row = ["<thead><tr><th class='sticky' rowspan='2'>Photo</th><th class='score-col' rowspan='2'>Score</th>"]
    field_row = ["</tr><tr>"]
    for group_name, fields in FIELD_GROUPS:
        group_row.append(f"<th class='group' colspan='{len(fields)}'>{esc(group_name)}</th>")
        field_row.extend(f"<th>{esc(FIELD_LABELS.get(field, field))}</th>" for field in fields)
    return "".join(group_row + field_row + ["</tr></thead>"])


def render_result_row(row: dict[str, Any], fields: list[str], images_dir: Path) -> str:
    scores = business_scores(row.get("expected") or {}, parsed(row))
    image = str(row["image"])
    image_path = (images_dir / image).resolve()
    cells = [
        "<tr>",
        "<td class='sticky'>",
        "<div class='photo'>",
        f"<img src='{esc(image_path.as_uri())}' alt='{esc(image)}'>",
        f"<div><b>{esc(image)}</b><span>{esc(status_text(row))}</span></div>",
        "</div>",
        "</td>",
        f"<td class='score-col'><b>{fmt_pct(scores['kik_business_score_pct'])}</b></td>",
    ]
    for field in fields:
        cells.append(render_value_cell(field, (row.get("expected") or {}).get(field), pred_value(row, field)))
    cells.append("</tr>")
    return "".join(cells)


def render_value_cell(field: str, expected: Any, predicted: Any) -> str:
    if expected is None:
        css = "na"
        score_label = "not scored"
    else:
        score = score_kik_value(field, expected, predicted)
        if score >= 1.0:
            css = "ok"
        elif score > 0:
            css = "partial"
        else:
            css = "bad"
        score_label = f"{score:.2f}"
    return (
        "<td>"
        f"<div class='cell {css}'>"
        f"<small>GT</small><strong>{esc(fmt_value(expected))}</strong>"
        f"<small>pred</small><strong>{esc(fmt_value(predicted))}</strong>"
        f"<small>score {esc(score_label)}</small>"
        "</div>"
        "</td>"
    )


def render_field_summary_row(field: str, rows: list[dict[str, Any]]) -> str:
    scores = []
    matched = 0
    scored = 0
    for row in rows:
        expected = (row.get("expected") or {}).get(field)
        if expected is None:
            continue
        scored += 1
        score = score_kik_value(field, expected, pred_value(row, field))
        scores.append(score)
        if score >= 1.0:
            matched += 1
    avg = sum(scores) / len(scores) if scores else None
    return (
        "<tr>"
        f"<td>{esc(FIELD_LABELS.get(field, field))}</td>"
        f"<td>{fmt_pct(avg * 100 if avg is not None else None)}</td>"
        f"<td>{matched}</td>"
        f"<td>{scored}</td>"
        "</tr>"
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    business = [business_scores(row.get("expected") or {}, parsed(row)) for row in rows]
    matched = 0
    scored = 0
    for row in rows:
        for field in BUSINESS_WEIGHTS:
            expected = (row.get("expected") or {}).get(field)
            if expected is None:
                continue
            scored += 1
            if score_kik_value(field, expected, pred_value(row, field)) >= 1.0:
                matched += 1
    return {
        "business_score": avg(score["kik_business_score_pct"] for score in business),
        "core_score": avg(score["core_kik_score_pct"] for score in business),
        "sku_score": avg(score["sku_family_score_pct"] for score in business),
        "execution_score": avg(score["execution_score_pct"] for score in business),
        "equipment_score": avg(score["equipment_photo_score_pct"] for score in business),
        "status_score": avg(score["status_actionability_score_pct"] for score in business),
        "matched_fields": matched,
        "scored_fields": scored,
    }


def metric(label: str, value: str) -> str:
    return f"<div class='metric'><span>{esc(label)}</span><b>{esc(value)}</b></div>"


def parsed(row: dict[str, Any]) -> dict[str, Any] | None:
    value = row.get("parsed")
    return value if isinstance(value, dict) else None


def pred_value(row: dict[str, Any], field: str) -> Any:
    value = parsed(row)
    return value.get(field) if value else None


def status_text(row: dict[str, Any]) -> str:
    if row.get("error"):
        return str(row["error"])
    return "schema ok" if row.get("schema_valid") else "schema invalid"


def fmt_value(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)


def fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}%"


def avg(values: Any) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def esc(value: str) -> str:
    return html.escape(value, quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
