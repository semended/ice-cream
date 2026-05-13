from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_eval.tasks.kik_simple.prompts import SYSTEM_PROMPT, USER_PROMPT, json_schema_instruction
from vlm_eval.tasks.kik_simple.schema import KIK_SIMPLE_REQUIRED_FIELDS

RUN_DIR_DEFAULT = Path("runs/kik_simple_gemma4_31b_full/20260511_181605")
IMAGES_DIR_DEFAULT = Path("data/real_images")
REFERENCE_SHEET_DEFAULT = Path("data/reference_images_sku_sheet/kik_sku_reference.png")

FIELD_LABELS = {
    "is_trade_equipment_photo": "Торговое оборудование",
    "is_ice_cream_equipment": "Оборудование с мороженым",
    "kik_present": "КИК присутствует",
    "kik_sku_count": "Уникальные SKU КИК",
    "kik_share_percent": "Доля КИК в верхней видимой части, %",
    "has_monobrand_block": "Монобрендовый блок",
    "has_non_icecream_products": "Посторонние не-мороженые товары",
    "is_kik_mixed_with_competitors": "КИК перемешан с конкурентами",
    "status_score": "Статус точки",
}

FIELD_GROUPS = [
    (
        "Фото и оборудование",
        [
            "is_trade_equipment_photo",
            "is_ice_cream_equipment",
        ],
    ),
    (
        "КИК и бизнес-итог",
        [
            "kik_present",
            "kik_sku_count",
            "kik_share_percent",
            "status_score",
        ],
    ),
    (
        "Выкладка и нарушения",
        [
            "has_monobrand_block",
            "has_non_icecream_products",
            "is_kik_mixed_with_competitors",
        ],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate photo-by-photo HTML audit report for kik_simple runs.")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR_DEFAULT)
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR_DEFAULT)
    parser.add_argument("--reference-sheet", type=Path, default=REFERENCE_SHEET_DEFAULT)
    parser.add_argument("--model-key", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.run_dir / "results.jsonl")
    if args.model_key:
        rows = [row for row in rows if row.get("model_key") == args.model_key]
    if not rows:
        raise SystemExit("No result rows found")
    rows.sort(key=lambda row: str(row.get("image") or ""))
    model_key = args.model_key or str(rows[0].get("model_key") or "model")
    summary = first_row(read_csv(args.run_dir / "summary.csv"), model_key)
    worst_by_image = {str(row.get("image")): row for row in read_csv(args.run_dir / "worst_cases_by_model.csv")}
    output = args.output or args.run_dir / f"{model_key}_simple_photo_audit.html"
    output.write_text(
        render_html(args.run_dir, args.images_dir, args.reference_sheet, output, model_key, rows, summary, worst_by_image),
        encoding="utf-8",
    )
    print(output)
    return 0


def render_html(
    run_dir: Path,
    images_dir: Path,
    reference_sheet: Path,
    output: Path,
    model_key: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    worst_by_image: dict[str, dict[str, Any]],
) -> str:
    exact_matches, exact_total = exact_totals(rows)
    parts = [
        "<!doctype html>",
        "<html lang='ru'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{esc(model_key)} kik_simple photo audit</title>",
        "<style>",
        css(),
        "</style>",
        "</head>",
        "<body>",
        "<main>",
        render_header(run_dir, model_key, rows, summary, exact_matches, exact_total),
        render_reference_and_prompt(reference_sheet, output),
        render_photo_sections(rows, images_dir, output, worst_by_image),
        "</main>",
        "</body>",
        "</html>",
    ]
    return "\n".join(parts)


def render_header(
    run_dir: Path,
    model_key: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    exact_matches: int,
    exact_total: int,
) -> str:
    exact_rate = exact_matches / exact_total * 100 if exact_total else 0
    error_count = sum(1 for row in rows if row.get("error"))
    return f"""
<header class="hero">
  <p class="eyebrow">KIK simple photo audit</p>
  <h1>{esc(model_key)}</h1>
  <p class="lead">Прогон <code>{esc(run_dir.as_posix())}</code>: одна SKU reference-sheet картинка, один target за запрос, без классификации по SKU-группам.</p>
  <div class="stats">
    {stat("Фото", len(rows))}
    {stat("Ошибки API/JSON", error_count)}
    {stat("Schema valid", fmt(summary.get("schema_valid_rate")))}
    {stat("Business score", fmt(summary.get("kik_simple_business_score_pct")))}
    {stat("Exact fields", f"{exact_matches}/{exact_total} ({exact_rate:.1f}%)")}
    {stat("SKU MAE", fmt(summary.get("kik_sku_count_mae")))}
    {stat("Share MAE", fmt(summary.get("kik_share_percent_mae")))}
  </div>
</header>
"""


def render_reference_and_prompt(reference_sheet: Path, output: Path) -> str:
    reference_src = rel_src(reference_sheet.resolve(), output)
    prompt_text = SYSTEM_PROMPT + "\n\n--- USER ---\n\n" + USER_PROMPT + "\n\n--- SCHEMA ---\n\n" + json_schema_instruction()
    return f"""
<section class="section">
  <div class="section-head">
    <p class="eyebrow">Reference + prompt</p>
    <h2>Что видела модель перед ответом</h2>
  </div>
  <div class="reference-layout">
    <figure class="reference-sheet">
      <img src="{esc(reference_src)}" alt="KIK SKU reference sheet">
      <figcaption>REF_SKU_SHEET: используется только как визуальный справочник SKU, не считается как target evidence.</figcaption>
    </figure>
    <details class="prompt" open>
      <summary>Prompt contract</summary>
      <pre>{esc(prompt_text)}</pre>
    </details>
  </div>
</section>
"""


def render_photo_sections(
    rows: list[dict[str, Any]],
    images_dir: Path,
    output: Path,
    worst_by_image: dict[str, dict[str, Any]],
) -> str:
    cards = [
        render_photo_section(index, row, images_dir, output, worst_by_image.get(str(row.get("image") or "")))
        for index, row in enumerate(rows, start=1)
    ]
    return f"""
<section class="section">
  <div class="section-head">
    <p class="eyebrow">Photo by photo</p>
    <h2>Фото, ответ модели и GT</h2>
  </div>
  <div class="photo-stack">{''.join(cards)}</div>
</section>
"""


def render_photo_section(
    index: int,
    row: dict[str, Any],
    images_dir: Path,
    output: Path,
    worst: dict[str, Any] | None,
) -> str:
    image = str(row.get("image") or "")
    image_src = rel_src((images_dir / image).resolve(), output)
    parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
    expected = row.get("expected") if isinstance(row.get("expected"), dict) else {}
    matches, total = row_exact_totals(row)
    business_score = (worst or {}).get("kik_simple_business_score_pct")
    error_fields = str((worst or {}).get("error_fields") or "")
    return f"""
<article class="photo-block" id="{esc(Path(image).stem)}">
  <div class="photo-head">
    <div>
      <p class="eyebrow">Фото {index}</p>
      <h3>{esc(image)}</h3>
    </div>
    <div class="photo-score">
      <strong>{fmt(business_score)}</strong>
      <span>business score</span>
    </div>
    <div class="photo-score">
      <strong>{matches}/{total}</strong>
      <span>exact fields</span>
    </div>
  </div>
  <div class="photo-layout">
    <figure class="target-photo">
      <img src="{esc(image_src)}" alt="{esc(image)}">
      <figcaption>{esc(image)}</figcaption>
    </figure>
    <div class="answers">
      {render_metric_groups(parsed, expected, row.get("field_scores") if isinstance(row.get("field_scores"), dict) else {})}
      {render_json_pair(parsed, expected, row)}
      {render_error_fields(error_fields)}
    </div>
  </div>
</article>
"""


def render_metric_groups(parsed: dict[str, Any], expected: dict[str, Any], field_scores: dict[str, Any]) -> str:
    return "".join(render_metric_group(title, fields, parsed, expected, field_scores) for title, fields in FIELD_GROUPS)


def render_metric_group(
    title: str,
    fields: list[str],
    parsed: dict[str, Any],
    expected: dict[str, Any],
    field_scores: dict[str, Any],
) -> str:
    rows = []
    for field in fields:
        pred = parsed.get(field)
        gt = expected.get(field)
        score = field_scores.get(field)
        if gt is None:
            row_class = "neutral"
            status = "GT null"
        elif pred == gt:
            row_class = "ok"
            status = "exact"
        else:
            row_class = "bad"
            status = "diff"
        rows.append(
            f"""
<tr class="{row_class}">
  <th>{esc(FIELD_LABELS.get(field, field))}<small>{esc(field)}</small></th>
  <td>{format_value(pred)}</td>
  <td>{format_value(gt)}</td>
  <td>{fmt(score) if score is not None else "n/a"}</td>
  <td>{esc(status)}</td>
</tr>
"""
        )
    return f"""
<section class="metric-group">
  <h4>{esc(title)}</h4>
  <table>
    <thead><tr><th>Поле</th><th>Модель</th><th>GT</th><th>Score</th><th>Итог</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""


def render_json_pair(parsed: dict[str, Any], expected: dict[str, Any], row: dict[str, Any]) -> str:
    raw = row.get("raw_response") or ""
    return f"""
<details class="json-dump">
  <summary>JSON ответа и GT</summary>
  <div class="json-grid">
    <div><h4>Parsed answer</h4><pre>{esc(json.dumps(parsed, ensure_ascii=False, indent=2))}</pre></div>
    <div><h4>Expected GT</h4><pre>{esc(json.dumps(expected, ensure_ascii=False, indent=2))}</pre></div>
  </div>
  <h4>Raw response</h4>
  <pre>{esc(str(raw))}</pre>
</details>
"""


def render_error_fields(error_fields: str) -> str:
    if not error_fields:
        return ""
    items = "".join(f"<span>{esc(field)}</span>" for field in error_fields.split(";") if field)
    return f"<div class='error-fields'><strong>Поля с ошибками:</strong>{items}</div>"


def exact_totals(rows: list[dict[str, Any]]) -> tuple[int, int]:
    matches = 0
    total = 0
    for row in rows:
        row_matches, row_total = row_exact_totals(row)
        matches += row_matches
        total += row_total
    return matches, total


def row_exact_totals(row: dict[str, Any]) -> tuple[int, int]:
    parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
    expected = row.get("expected") if isinstance(row.get("expected"), dict) else {}
    fields = [field for field in KIK_SIMPLE_REQUIRED_FIELDS if expected.get(field) is not None]
    matches = sum(1 for field in fields if parsed.get(field) == expected.get(field))
    return matches, len(fields)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: coerce(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def first_row(rows: list[dict[str, Any]], model_key: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get("model_key")) == model_key:
            return row
    return rows[0] if rows else {}


def coerce(value: str | None) -> Any:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def format_value(value: Any) -> str:
    if value is None:
        return "<span class='null'>null</span>"
    if value is True:
        return "<span class='bool true'>true</span>"
    if value is False:
        return "<span class='bool false'>false</span>"
    if isinstance(value, (dict, list)):
        return f"<code>{esc(json.dumps(value, ensure_ascii=False))}</code>"
    return esc(str(value))


def stat(label: str, value: Any) -> str:
    return f"<span><strong>{esc(str(value))}</strong>{esc(label)}</span>"


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def rel_src(path: Path, output: Path) -> str:
    return os.path.relpath(path, output.parent.resolve()).replace(os.sep, "/")


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def css() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f6f7f5;
  --panel: #ffffff;
  --ink: #18201c;
  --muted: #66736c;
  --line: #d9dfd9;
  --blue: #1f5f9a;
  --green-bg: #e4f4e7;
  --green-line: #8cc998;
  --green-ink: #176334;
  --red-bg: #fae4e2;
  --red-line: #dc9790;
  --red-ink: #982d25;
  --neutral-bg: #eef2f6;
  --neutral-line: #b9c7d6;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main { width: min(1500px, calc(100% - 36px)); margin: 0 auto; padding: 28px 0 60px; }
.hero { border-bottom: 1px solid var(--line); padding: 12px 0 24px; }
.eyebrow { margin: 0 0 7px; color: var(--blue); text-transform: uppercase; font-size: 11px; font-weight: 850; letter-spacing: .08em; }
h1 { margin: 0; font-size: clamp(36px, 6vw, 72px); line-height: .95; letter-spacing: 0; }
h2 { margin: 0; font-size: 26px; letter-spacing: 0; }
h3 { margin: 0; font-size: 24px; letter-spacing: 0; }
h4 { margin: 0 0 8px; font-size: 15px; letter-spacing: 0; }
.lead { max-width: 980px; margin: 14px 0 0; color: var(--muted); font-size: 16px; }
code { background: #edf0ed; border-radius: 5px; padding: 2px 5px; }
.stats { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 18px; }
.stats span {
  display: inline-flex;
  gap: 8px;
  align-items: baseline;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  padding: 9px 11px;
  color: var(--muted);
}
.stats strong { color: var(--ink); font-size: 19px; }
.section { margin-top: 30px; }
.section-head { display: flex; justify-content: space-between; align-items: end; gap: 20px; margin-bottom: 14px; }
.reference-layout { display: grid; grid-template-columns: minmax(380px, .9fr) 1.1fr; gap: 14px; align-items: start; }
.reference-sheet, .photo-block, .prompt, .metric-group, .json-dump {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}
.reference-sheet { margin: 0; padding: 12px; }
.reference-sheet img { width: 100%; display: block; border-radius: 6px; background: #fff; }
figcaption { margin-top: 8px; color: var(--muted); font-size: 12px; }
.prompt { padding: 12px; }
.prompt summary, .json-dump summary { cursor: pointer; font-weight: 800; }
pre {
  margin: 10px 0 0;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  color: #24302a;
  background: #f2f4f2;
  border: 1px solid var(--line);
  border-radius: 7px;
  padding: 10px;
  font-size: 12px;
  max-height: 420px;
  overflow: auto;
}
.photo-stack { display: grid; gap: 16px; }
.photo-block { padding: 14px; }
.photo-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 12px; }
.photo-score {
  min-width: 116px;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 8px 10px;
  text-align: right;
}
.photo-score strong { display: block; font-size: 21px; }
.photo-score span { color: var(--muted); font-size: 12px; }
.photo-layout { display: grid; grid-template-columns: minmax(360px, .85fr) 1.15fr; gap: 14px; align-items: start; }
.target-photo { margin: 0; position: sticky; top: 14px; }
.target-photo img {
  width: 100%;
  display: block;
  max-height: 760px;
  object-fit: contain;
  border-radius: 8px;
  background: #f0f2f0;
  border: 1px solid var(--line);
}
.answers { display: grid; gap: 12px; }
.metric-group { overflow: hidden; }
.metric-group h4 { padding: 11px 12px 0; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 8px 9px; border-top: 1px solid var(--line); vertical-align: top; text-align: left; }
th { width: 31%; font-weight: 760; }
th small { display: block; margin-top: 2px; color: var(--muted); font-weight: 520; }
tr.ok { background: var(--green-bg); }
tr.ok td:last-child { color: var(--green-ink); font-weight: 780; }
tr.bad { background: var(--red-bg); }
tr.bad td:last-child { color: var(--red-ink); font-weight: 780; }
tr.neutral { background: var(--neutral-bg); }
.null { color: #7c8790; font-style: italic; }
.bool.true { color: var(--green-ink); font-weight: 750; }
.bool.false { color: #6f3b18; font-weight: 750; }
.json-dump { padding: 12px; }
.json-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.error-fields {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 7px;
  color: var(--muted);
}
.error-fields span {
  background: var(--red-bg);
  color: var(--red-ink);
  border: 1px solid var(--red-line);
  border-radius: 999px;
  padding: 4px 8px;
  font-size: 12px;
}
@media (max-width: 980px) {
  main { width: min(100% - 22px, 1500px); }
  .reference-layout, .photo-layout, .json-grid { grid-template-columns: 1fr; }
  .target-photo { position: static; }
  .section-head, .photo-head { align-items: start; flex-direction: column; }
  .photo-score { text-align: left; }
}
"""


if __name__ == "__main__":
    raise SystemExit(main())
