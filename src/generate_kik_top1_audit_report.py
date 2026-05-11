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

from vlm_eval.tasks.kik.prompts import SYSTEM_PROMPT, USER_PROMPT, json_schema_instruction
from vlm_eval.tasks.kik.schema import KIK_REQUIRED_FIELDS


RUN_DIR_DEFAULT = Path("runs/kik_eval_ref_7x10/20260510_214604")
OUTPUT_DIR_DEFAULT = Path("runs/kik_eval_ref_7x10_top1_audit/20260510_214604_qwen25_vl_72b")
IMAGES_DIR_DEFAULT = Path("data/real_images")
REFERENCES_DIR_DEFAULT = Path("data/reference_images")
EXCLUDE_FIELDS_DEFAULT: tuple[str, ...] = ()

FIELD_LABELS = {
    "is_trade_equipment_photo": "Торговое оборудование",
    "is_ice_cream_equipment": "Оборудование с мороженым",
    "photo_crop_is_full": "Ларь полностью в кадре",
    "kik_present": "КИК присутствует",
    "kik_sku_count": "SKU КИК, шт.",
    "kik_share_percent": "Доля КИК, %",
    "has_cup": "Стакан",
    "has_eskimo": "Эскимо",
    "has_lakomka": "Лакомка",
    "has_cone": "Рожок",
    "has_sandwich": "Сэндвич",
    "has_bucket": "Ведро",
    "has_poleno_or_briquette": "Брикет или полено",
    "has_posm": "POSM / фирменные ценники",
    "has_monobrand_block": "Монобрендовый блок",
    "has_foreign_label": "Чужая бирка",
    "has_non_icecream_products": "Посторонние товары",
    "has_empty_sections": "Пустые секции",
    "is_kik_mixed_with_competitors": "КИК перемешан с конкурентами",
    "status_score": "Статус точки",
}

FIELD_GROUPS = [
    (
        "Фото и оборудование",
        [
            "is_trade_equipment_photo",
            "is_ice_cream_equipment",
            "photo_crop_is_full",
        ],
    ),
    (
        "КИК и объем",
        [
            "kik_present",
            "kik_sku_count",
            "kik_share_percent",
            "status_score",
        ],
    ),
    (
        "SKU-группы",
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
        "Выкладка и нарушения",
        [
            "has_posm",
            "has_monobrand_block",
            "has_foreign_label",
            "has_non_icecream_products",
            "has_empty_sections",
            "is_kik_mixed_with_competitors",
        ],
    ),
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate photo-by-photo audit report for the top KIK VLM model.")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR_DEFAULT)
    parser.add_argument("--model-key", type=str, default=None)
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR_DEFAULT)
    parser.add_argument("--references-dir", type=Path, default=REFERENCES_DIR_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--exclude-fields", type=str, default=",".join(EXCLUDE_FIELDS_DEFAULT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = read_csv(args.run_dir / "summary.csv")
    model_key = args.model_key or top_model_key(summaries)
    rows = [
        row
        for row in read_jsonl(args.run_dir / "results.jsonl")
        if row.get("model_key") == model_key
    ]
    if not rows:
        raise SystemExit(f"No rows for model: {model_key}")
    rows.sort(key=lambda row: str(row.get("image") or ""))
    exclude_fields = {field.strip() for field in args.exclude_fields.split(",") if field.strip()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{model_key}_photo_audit.html"
    output.write_text(
        render_html(args.run_dir, args.images_dir, args.references_dir, output, model_key, rows, exclude_fields),
        encoding="utf-8",
    )
    print(output)
    return 0


def top_model_key(summaries: list[dict[str, Any]]) -> str:
    valid = [row for row in summaries if not run_failed(row)]
    rows = valid or summaries
    best = max(rows, key=lambda row: as_float(row.get("kik_business_score_pct")) or -1.0)
    return str(best["model_key"])


def run_failed(row: dict[str, Any]) -> bool:
    errors = as_float(row.get("api_or_json_errors")) or 0.0
    total = as_float(row.get("total_cases")) or 0.0
    schema = as_float(row.get("schema_valid_rate")) or 0.0
    return bool(total and errors / total > 0.5) or schema < 0.5


def render_html(
    run_dir: Path,
    images_dir: Path,
    references_dir: Path,
    output: Path,
    model_key: str,
    rows: list[dict[str, Any]],
    exclude_fields: set[str],
) -> str:
    total_matches, total_fields = match_totals(rows, exclude_fields)
    parts = [
        "<!doctype html>",
        "<html lang='ru'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{esc(model_key)} KIK audit</title>",
        "<style>",
        css(),
        "</style>",
        "</head>",
        "<body>",
        "<main>",
        render_header(run_dir, model_key, rows, total_matches, total_fields, exclude_fields),
        render_prompt(),
        render_references(references_dir, output),
        render_photo_sections(rows, images_dir, output, exclude_fields),
        "</main>",
        "</body>",
        "</html>",
    ]
    return "\n".join(parts)


def render_header(
    run_dir: Path,
    model_key: str,
    rows: list[dict[str, Any]],
    total_matches: int,
    total_fields: int,
    exclude_fields: set[str],
) -> str:
    accuracy = total_matches / total_fields * 100 if total_fields else 0
    filtered_note = ""
    if exclude_fields:
        filtered_note = f"<p class='filter-note'>Фильтр: скрыто {len(exclude_fields)} полей, счетчик пересчитан без них.</p>"
    return f"""
<header class="hero">
  <p class="eyebrow">Top-1 model audit</p>
  <h1>{esc(model_key)}</h1>
  <p class="lead">Разбор прогона <code>{esc(run_dir.as_posix())}</code>: prompt, reference images и 10 фото с построчным сравнением ответа модели против GT.</p>
  <div class="hero-stats">
    <span><strong>{len(rows)}</strong> фото</span>
    <span><strong>{total_matches}/{total_fields}</strong> strict matches</span>
    <span><strong>{accuracy:.1f}%</strong> exact field accuracy</span>
  </div>
  {filtered_note}
</header>
"""


def render_prompt() -> str:
    return f"""
<section class="section">
  <div class="section-head">
    <p class="eyebrow">1. Prompt</p>
    <h2>Что получила модель</h2>
  </div>
  <div class="prompt-grid">
    <article>
      <h3>System prompt</h3>
      <pre>{esc(SYSTEM_PROMPT)}</pre>
    </article>
    <article>
      <h3>User prompt + schema instruction</h3>
      <pre>{esc(USER_PROMPT + chr(10) + chr(10) + json_schema_instruction())}</pre>
    </article>
  </div>
</section>
"""


def render_references(references_dir: Path, output: Path) -> str:
    refs = [
        path
        for path in sorted(references_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    cards = []
    for path in refs:
        src = rel_src(path.resolve(), output)
        cards.append(
            f"""
<figure class="ref-card">
  <img src="{esc(src)}" alt="{esc(path.stem)}">
  <figcaption>{esc(path.stem.replace('_', ' '))}</figcaption>
</figure>
"""
        )
    return f"""
<section class="section">
  <div class="section-head">
    <p class="eyebrow">2. References</p>
    <h2>Референсы SKU, которые были в прогоне</h2>
  </div>
  <div class="refs">{''.join(cards)}</div>
</section>
"""


def render_photo_sections(
    rows: list[dict[str, Any]],
    images_dir: Path,
    output: Path,
    exclude_fields: set[str],
) -> str:
    cards = []
    for index, row in enumerate(rows, start=1):
        cards.append(render_photo_section(index, row, images_dir, output, exclude_fields))
    return f"""
<section class="section">
  <div class="section-head">
    <p class="eyebrow">3. Photo by photo</p>
    <h2>10 отдельных блоков: фото, ответ модели, GT</h2>
  </div>
  <div class="photo-stack">{''.join(cards)}</div>
</section>
"""


def render_photo_section(
    index: int,
    row: dict[str, Any],
    images_dir: Path,
    output: Path,
    exclude_fields: set[str],
) -> str:
    image = str(row.get("image") or "")
    image_src = rel_src((images_dir / image).resolve(), output)
    parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
    expected = row.get("expected") if isinstance(row.get("expected"), dict) else {}
    matches, total = row_match_totals(row, exclude_fields)
    table_parts = []
    for group_title, fields in FIELD_GROUPS:
        visible_fields = [field for field in fields if field not in exclude_fields]
        if visible_fields:
            table_parts.append(render_metric_group(group_title, visible_fields, parsed, expected))
    return f"""
<article class="photo-block" id="{esc(Path(image).stem)}">
  <div class="photo-head">
    <div>
      <p class="eyebrow">Фото {index}</p>
      <h3>{esc(image)}</h3>
    </div>
    <strong>{matches}/{total} совпало</strong>
  </div>
  <div class="photo-layout">
    <figure class="target-photo">
      <img src="{esc(image_src)}" alt="{esc(image)}">
      <figcaption>{esc(image)}</figcaption>
    </figure>
    <div class="metric-tables">
      {''.join(table_parts)}
    </div>
  </div>
</article>
"""


def render_metric_group(
    title: str,
    fields: list[str],
    parsed: dict[str, Any],
    expected: dict[str, Any],
) -> str:
    rows = []
    for field in fields:
        prediction = parsed.get(field)
        gt = expected.get(field)
        ok = values_equal(prediction, gt)
        rows.append(
            "<tr class='" + ("ok" if ok else "bad") + "'>"
            f"<th>{esc(FIELD_LABELS.get(field, field))}<small>{esc(field)}</small></th>"
            f"<td>{format_value(prediction)}</td>"
            f"<td>{format_value(gt)}</td>"
            f"<td>{'совпало' if ok else 'не совпало'}</td>"
            "</tr>"
        )
    return f"""
<section class="metric-group">
  <h4>{esc(title)}</h4>
  <table>
    <thead><tr><th>Метрика</th><th>Ответ модели</th><th>GT</th><th>Итог</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>
"""

def match_totals(rows: list[dict[str, Any]], exclude_fields: set[str]) -> tuple[int, int]:
    matches = 0
    total = 0
    for row in rows:
        row_matches, row_total = row_match_totals(row, exclude_fields)
        matches += row_matches
        total += row_total
    return matches, total


def row_match_totals(row: dict[str, Any], exclude_fields: set[str]) -> tuple[int, int]:
    parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
    expected = row.get("expected") if isinstance(row.get("expected"), dict) else {}
    fields = [field for field in KIK_REQUIRED_FIELDS if field not in exclude_fields]
    matches = sum(1 for field in fields if values_equal(parsed.get(field), expected.get(field)))
    return matches, len(fields)


def values_equal(left: Any, right: Any) -> bool:
    return left == right


def format_value(value: Any) -> str:
    if value is None:
        return "<span class='null'>null</span>"
    if value is True:
        return "<span>true</span>"
    if value is False:
        return "<span>false</span>"
    if isinstance(value, (dict, list)):
        return f"<code>{esc(json.dumps(value, ensure_ascii=False))}</code>"
    return esc(str(value))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: coerce(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def coerce(value: str | None) -> Any:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def rel_src(path: Path, output: Path) -> str:
    return os.path.relpath(path, output.parent.resolve()).replace(os.sep, "/")


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def css() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f4f2ed;
  --ink: #1d2420;
  --muted: #626d66;
  --line: #d7d2c7;
  --panel: #fffefa;
  --green-bg: #dcf1e3;
  --green-line: #79b98b;
  --green-ink: #17623b;
  --red-bg: #f7dddd;
  --red-line: #d38a8a;
  --red-ink: #9b2f2f;
  --blue: #2d6384;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main { width: min(1440px, calc(100% - 40px)); margin: 0 auto; padding: 32px 0 64px; }
.hero { padding: 24px 0 26px; border-bottom: 1px solid var(--line); }
.eyebrow { margin: 0 0 8px; color: var(--blue); text-transform: uppercase; font-size: 12px; font-weight: 850; letter-spacing: .08em; }
h1 { margin: 0; font-size: clamp(42px, 7vw, 84px); line-height: .95; letter-spacing: 0; }
h2 { margin: 0; font-size: 28px; letter-spacing: 0; }
h3 { margin: 0; font-size: 28px; letter-spacing: 0; }
h4 { margin: 0 0 10px; font-size: 16px; letter-spacing: 0; }
.lead { max-width: 980px; margin: 16px 0 0; color: var(--muted); font-size: 17px; }
code { background: #e8e4db; border-radius: 5px; padding: 2px 5px; }
.hero-stats { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }
.hero-stats span {
  display: inline-flex;
  gap: 8px;
  align-items: baseline;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  padding: 10px 12px;
  color: var(--muted);
}
.hero-stats strong { color: var(--ink); font-size: 20px; }
.filter-note { max-width: 980px; margin: 14px 0 0; color: var(--muted); }
.section { margin-top: 34px; }
.section-head { display: flex; justify-content: space-between; align-items: end; gap: 24px; margin-bottom: 14px; }
.prompt-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.prompt-grid article, .photo-block {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}
.prompt-grid article { padding: 16px; min-width: 0; }
pre {
  margin: 12px 0 0;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  color: #2f3733;
  font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.refs { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; }
.ref-card {
  margin: 0;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  overflow: hidden;
}
.ref-card img { width: 100%; aspect-ratio: 1 / 1; object-fit: contain; display: block; background: #ece9e1; }
figcaption { padding: 8px 10px; color: var(--muted); font-size: 12px; }
.photo-stack { display: grid; gap: 18px; }
.photo-block { padding: 16px; }
.photo-head { display: flex; justify-content: space-between; align-items: start; gap: 20px; margin-bottom: 14px; }
.photo-head strong {
  flex: 0 0 auto;
  border-radius: 999px;
  background: #e8e4db;
  padding: 8px 12px;
  font-size: 16px;
}
.photo-layout { display: grid; grid-template-columns: minmax(300px, 420px) 1fr; gap: 16px; align-items: start; }
.target-photo { margin: 0; position: sticky; top: 12px; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: #ece9e1; }
.target-photo img { width: 100%; max-height: 720px; object-fit: contain; display: block; }
.metric-tables { display: grid; gap: 12px; min-width: 0; }
.metric-group { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: #fff; }
.metric-group h4 { padding: 12px; border-bottom: 1px solid var(--line); background: #eeebe3; }
table { width: 100%; border-collapse: collapse; table-layout: fixed; }
th, td { padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; overflow-wrap: anywhere; }
thead th { color: var(--muted); font-size: 12px; text-transform: uppercase; }
tbody tr:last-child th, tbody tr:last-child td { border-bottom: 0; }
tbody th { width: 32%; }
tbody th small { display: block; margin-top: 2px; color: var(--muted); font-weight: 500; }
tbody td:last-child { width: 120px; font-weight: 800; }
tr.ok th, tr.ok td { background: var(--green-bg); border-color: var(--green-line); color: var(--green-ink); }
tr.bad th, tr.bad td { background: var(--red-bg); border-color: var(--red-line); color: var(--red-ink); }
.null { color: #6d655b; font-style: italic; }
.notes { padding-bottom: 12px; }
.notes-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 0 12px; }
.notes-grid span { display: block; color: var(--muted); font-weight: 800; margin-bottom: 6px; }
.notes-grid p, .notes-grid ul { margin: 0; }
.notes-grid ul { padding-left: 18px; }
@media (max-width: 1100px) {
  .prompt-grid { grid-template-columns: 1fr; }
  .refs { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .photo-layout { grid-template-columns: 1fr; }
  .target-photo { position: static; }
}
@media (max-width: 700px) {
  main { width: min(100% - 24px, 1440px); padding-top: 18px; }
  .section-head, .photo-head { display: block; }
  .photo-head strong { display: inline-block; margin-top: 10px; }
  .refs { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .notes-grid { grid-template-columns: 1fr; }
  th, td { padding: 8px; }
}
"""


if __name__ == "__main__":
    raise SystemExit(main())
