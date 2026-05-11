from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_eval.tasks.kik.prompts import SYSTEM_PROMPT, USER_PROMPT, json_schema_instruction


RUN_DIR_DEFAULT = Path("runs/kik_eval_7x10_merged/20260510_142125")
IMAGES_DIR_DEFAULT = Path("data/real_images")

FIELD_GROUPS = [
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
    (
        "Фото и оборудование",
        [
            "is_trade_equipment_photo",
            "is_ice_cream_equipment",
            "photo_crop_is_full",
        ],
    ),
]

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

BOOLEAN_FIELDS = {
    field
    for _, fields in FIELD_GROUPS
    for field in fields
    if not field.endswith("_score") and not field.endswith("_percent") and field != "kik_sku_count"
}

SCORE_FIELDS_FOR_ROW = [field for _, fields in FIELD_GROUPS for field in fields]

BUSINESS_WEIGHTS = {
    "kik_present": 12.0,
    "kik_sku_count": 12.0,
    "kik_share_percent": 12.0,
    "has_cup": 4.0,
    "has_eskimo": 4.0,
    "has_lakomka": 4.0,
    "has_cone": 3.0,
    "has_sandwich": 3.0,
    "has_bucket": 2.0,
    "has_poleno_or_briquette": 3.5,
    "has_monobrand_block": 4.0,
    "is_kik_mixed_with_competitors": 4.0,
    "has_posm": 3.0,
    "has_foreign_label": 1.5,
    "has_non_icecream_products": 1.5,
    "has_empty_sections": 1.0,
    "is_trade_equipment_photo": 2.0,
    "is_ice_cream_equipment": 2.0,
    "photo_crop_is_full": 0.75,
    "status_score": 8.0,
}

MARKET_INFO = {
    "qwen25_vl_72b": {
        "price_in": 0.25,
        "price_out": 0.75,
        "context": "32k",
        "size": "72B dense",
        "license": "Apache-2.0, проверить exact checkpoint перед договором",
        "server_fit": "Реально, но тяжелее 30B: 1x80GB low-bit для старта, 2-4x80GB для запаса.",
        "fit_level": "good",
        "source": "https://openrouter.ai/qwen/qwen2.5-vl-72b-instruct",
    },
    "mistral_large_3": {
        "price_in": 0.50,
        "price_out": 1.50,
        "context": "262k",
        "size": "675B total / 41B active",
        "license": "Apache-2.0",
        "server_fit": "Benchmark/API/кластер. Для своего сервера слишком большой как первый prod-кандидат.",
        "fit_level": "benchmark",
        "source": "https://openrouter.ai/mistralai/mistral-large-2512",
    },
    "qwen3_vl_30b": {
        "price_in": 0.13,
        "price_out": 0.52,
        "context": "131k",
        "size": "30B total / 3B active",
        "license": "Apache-2.0, проверить exact checkpoint перед договором",
        "server_fit": "Самый реалистичный self-host: 1x80GB комфортно, 24-48GB в low-bit для прототипа.",
        "fit_level": "good",
        "source": "https://openrouter.ai/qwen/qwen3-vl-30b-a3b-instruct",
    },
    "gemma4_31b": {
        "price_in": 0.13,
        "price_out": 0.38,
        "context": "262k",
        "size": "30.7B dense",
        "license": "Apache-2.0",
        "server_fit": "Реально на 1x80GB или low-bit 24-48GB, но текущая latency/quality не радует.",
        "fit_level": "medium",
        "source": "https://openrouter.ai/google/gemma-4-31b-it",
    },
    "glm_46v": {
        "price_in": 0.30,
        "price_out": 0.90,
        "context": "131k",
        "size": "106B class",
        "license": "MIT",
        "server_fit": "Скорее hosted или 2x80GB self-host. Хорош как fallback на share/execution.",
        "fit_level": "medium",
        "source": "https://openrouter.ai/z-ai/glm-4.6v",
    },
    "qwen3_vl_235b": {
        "price_in": 0.20,
        "price_out": 0.88,
        "context": "262k",
        "size": "235B total / 22B active",
        "license": "Проверить exact checkpoint перед договором",
        "server_fit": "Benchmark only: 4-8 GPU class, экономика хуже 30B/72B.",
        "fit_level": "benchmark",
        "source": "https://openrouter.ai/qwen/qwen3-vl-235b-a22b-instruct",
    },
    "mistral_small_4": {
        "price_in": 0.15,
        "price_out": 0.60,
        "context": "262k",
        "size": "119B total / 6.5B active",
        "license": "Apache-2.0",
        "server_fit": "Технически реально на 2x80GB, но по этому тесту качество слишком низкое.",
        "fit_level": "weak",
        "source": "https://openrouter.ai/mistralai/mistral-small-2603",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate executive KIK VLM benchmark HTML report.")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR_DEFAULT)
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR_DEFAULT)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    output = args.output or run_dir / "kik_executive_model_report.html"
    results = read_jsonl(run_dir / "results.jsonl")
    summaries = read_csv(run_dir / "summary.csv")
    summaries = sorted(summaries, key=lambda row: as_float(row.get("kik_business_score_pct")) or -1, reverse=True)
    rank_map = {row["model_key"]: index + 1 for index, row in enumerate(summaries)}
    by_model = group_results(results)
    html_text = render_report(run_dir, args.images_dir, output, summaries, by_model, rank_map)
    output.write_text(html_text, encoding="utf-8")
    print(output)
    return 0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append({key: coerce(value) for key, value in row.items()})
        return rows


def coerce(value: str | None) -> Any:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def group_results(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["model_key"]].append(result)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["image"])
    return grouped


def render_report(
    run_dir: Path,
    images_dir: Path,
    output: Path,
    summaries: list[dict[str, Any]],
    by_model: dict[str, list[dict[str, Any]]],
    rank_map: dict[str, int],
) -> str:
    parts = [
        "<!doctype html>",
        "<html lang='ru'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>KIK VLM model report</title>",
        "<style>",
        css(),
        "</style>",
        "</head>",
        "<body>",
        render_hero(run_dir, summaries),
        render_model_nav(summaries),
        "<main>",
        render_contract(run_dir),
        render_ranking(summaries, by_model),
        render_recommendation(summaries),
    ]
    for summary in summaries:
        model_key = str(summary["model_key"])
        parts.append(render_model_section(summary, by_model[model_key], rank_map[model_key], images_dir, output))
    parts.extend(
        [
            render_sources(),
            "</main>",
            "</body>",
            "</html>",
        ]
    )
    return "\n".join(parts)


def render_hero(run_dir: Path, summaries: list[dict[str, Any]]) -> str:
    best = summaries[0]
    return f"""
<header class="top">
  <div class="top__copy">
    <p class="eyebrow">KIK retail execution VLM benchmark</p>
    <h1>7 моделей, 10 фото: кто реально видит КИК, SKU и выкладку</h1>
    <p class="lead">Отчет по прогону <span class="mono">{esc(run_dir.as_posix())}</span>. Модели отсортированы по <strong>kik_business_score_pct</strong>: от лучшей в зачете к худшей. Абсолютный уровень низкий у всех, поэтому это рейтинг кандидатов, а не зеленый свет на полностью автоматический prod.</p>
  </div>
  <div class="hero-score">
    <span>Текущий лидер</span>
    <strong>#{esc(str(best["model_key"]))}</strong>
    <b>{fmt_pct(best.get("kik_business_score_pct"))}</b>
    <em>business score</em>
  </div>
</header>
"""


def render_model_nav(summaries: list[dict[str, Any]]) -> str:
    links = []
    for index, row in enumerate(summaries, start=1):
        links.append(
            f"<a href='#{anchor(row['model_key'])}'><span>#{index}</span>{esc(row['model_key'])}<b>{fmt_pct(row.get('kik_business_score_pct'))}</b></a>"
        )
    return "<nav class='model-nav'>" + "".join(links) + "</nav>"


def render_contract(run_dir: Path) -> str:
    config_path = run_dir / "config_snapshot.yaml"
    return f"""
<section class="section contract">
  <div class="section-head">
    <p class="eyebrow">Что подавали VLM</p>
    <h2>Prompt, input contract и критерии оценки</h2>
  </div>
  <div class="contract-grid">
    <article class="plain-panel">
      <h3>Вход в каждый запрос</h3>
      <ul>
        <li>Одна target-фотография из <span class="mono">data/real_images</span>; в папке лежат только реальные JPG.</li>
        <li>Изображение кодировалось как <span class="mono">image_url</span> и ужималось до max side 1024 px по config.</li>
        <li>Reference-картинки SKU в этом isolated-run <strong>не отправлялись</strong>: модель видела только target photo, system prompt, user prompt и JSON schema.</li>
        <li>Температура 0, output limit 512 токенов у всех, кроме GLM с 1024 токенами после фикса.</li>
      </ul>
      <p class="small">Config snapshot: <span class="mono">{esc(config_path.as_posix())}</span></p>
    </article>
    <article class="plain-panel">
      <h3>Scoring</h3>
      <ul>
        <li>GT: ручная разметка <span class="mono">data/ground_truth/manual_ground_truth.jsonl</span>.</li>
        <li>Boolean/status поля: exact match. Numeric поля: частичный credit по расстоянию до GT.</li>
        <li>Business score: взвешенная сумма по КИК presence, SKU count, share, fill, SKU families, execution, photo/equipment и status.</li>
        <li>Цвета: green = совпало, yellow = частично, red = ошибка, gray = GT/prediction нет или поле не скорится.</li>
      </ul>
    </article>
  </div>
  <details class="prompt-box">
    <summary>Показать system prompt</summary>
    <pre>{esc(SYSTEM_PROMPT)}</pre>
  </details>
  <details class="prompt-box">
    <summary>Показать user prompt + schema instruction</summary>
    <pre>{esc(USER_PROMPT + chr(10) + chr(10) + json_schema_instruction())}</pre>
  </details>
</section>
"""


def render_ranking(summaries: list[dict[str, Any]], by_model: dict[str, list[dict[str, Any]]]) -> str:
    rows = []
    for index, row in enumerate(summaries, start=1):
        model_key = str(row["model_key"])
        cost = cost_stats(model_key, by_model[model_key])
        capacity = capacity_stats(row)
        market = MARKET_INFO.get(model_key, {})
        rows.append(
            "<tr>"
            f"<td class='rank-cell'>#{index}</td>"
            f"<td><a href='#{anchor(model_key)}'>{esc(model_key)}</a><span>{esc(str(row.get('model') or ''))}</span></td>"
            f"<td>{fmt_pct(row.get('kik_business_score_pct'))}</td>"
            f"<td>{fmt_pct(row.get('core_kik_score_pct'))}</td>"
            f"<td>{fmt_pct(row.get('sku_family_score_pct'))}</td>"
            f"<td>{fmt_pct(row.get('execution_score_pct'))}</td>"
            f"<td>{fmt_num(row.get('kik_present_f1'), 2)}</td>"
            f"<td>{fmt_num(row.get('kik_sku_count_mae'), 1)}</td>"
            f"<td>{fmt_num(row.get('kik_share_percent_mae'), 1)} pp</td>"
            f"<td>{fmt_seconds(row.get('p95_latency_sec'))}</td>"
            f"<td>{fmt_money(cost['cost_per_1k'])}/1k</td>"
            f"<td>{esc(capacity['workers_for_50k'])}</td>"
            f"<td><span class='fit fit-{esc(str(market.get('fit_level', 'unknown')))}'>{esc(str(market.get('server_fit', 'нет данных')))}</span></td>"
            "</tr>"
        )
    return f"""
<section class="section">
  <div class="section-head">
    <p class="eyebrow">Зачет</p>
    <h2>Рейтинг моделей от топа к хвосту</h2>
  </div>
  <div class="table-wrap">
    <table class="ranking">
      <thead>
        <tr>
          <th>#</th><th>Модель</th><th>Business</th><th>Core KIK</th><th>SKU family</th><th>Execution</th><th>KIK F1</th><th>SKU MAE</th><th>Share MAE</th><th>p95</th><th>Cost</th><th>Workers 50k/day*</th><th>Server reality</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
  <p class="footnote">*Грубая оценка по observed avg latency одного запроса: сколько параллельных потоков нужно для 50k фото/сутки. Это не заменяет нагрузочный тест на своем GPU.</p>
</section>
"""


def render_recommendation(summaries: list[dict[str, Any]]) -> str:
    best = summaries[0]
    return f"""
<section class="section recommendation">
  <div class="section-head">
    <p class="eyebrow">Вывод</p>
    <h2>Кого ставить в prod</h2>
  </div>
  <div class="rec-layout">
    <div>
      <h3>Оптимальный кандидат: Qwen2.5-VL-72B-Instruct</h3>
      <p>По этому прогону он первый в зачете: <strong>{fmt_pct(best.get('kik_business_score_pct'))}</strong> business score, лучший баланс SKU family и execution среди self-host кандидатов. Он тяжелее Qwen3-VL-30B, зато сейчас дает заметно более полезный прикладной сигнал.</p>
      <p>Но важное ограничение: это <strong>не готово для полностью автоматического prod</strong>. У лидера KIK F1 всего {fmt_num(best.get('kik_present_f1'), 2)}, SKU MAE {fmt_num(best.get('kik_sku_count_mae'), 1)}, share MAE {fmt_num(best.get('kik_share_percent_mae'), 1)} pp, critical recall {fmt_num(best.get('critical_recall'), 2)}. Для боевого запуска нужен режим manual-assist/fallback, доработка prompt/GT и повторный eval на большем golden set.</p>
    </div>
    <div class="rec-aside">
      <strong>Практическая схема</strong>
      <p>Primary: Qwen2.5-VL-72B. Fallback/second pass: GLM-4.6V на спорных share/execution кейсах. Benchmark: Mistral Large 3 только для разбора ошибок. Qwen3-VL-30B держать как дешевый self-host путь после prompt tuning.</p>
    </div>
  </div>
</section>
"""


def render_model_section(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    rank: int,
    images_dir: Path,
    output: Path,
) -> str:
    model_key = str(summary["model_key"])
    market = MARKET_INFO.get(model_key, {})
    cost = cost_stats(model_key, rows)
    capacity = capacity_stats(summary)
    verdict = model_verdict(model_key, summary)
    photo_cards = "\n".join(render_photo_card(row, images_dir, output) for row in rows)
    return f"""
<section class="section model-section" id="{anchor(model_key)}">
  <div class="model-head">
    <div>
      <p class="eyebrow">#{rank} / {esc(str(summary.get('role') or ''))}</p>
      <h2>{esc(model_key)}</h2>
      <p class="model-name">{esc(str(summary.get('model') or ''))}</p>
    </div>
    <div class="big-score">{fmt_pct(summary.get('kik_business_score_pct'))}</div>
  </div>
  <div class="model-dashboard">
    {metric_tile('Стоимость run', fmt_money(cost['run_cost']), f"{fmt_money(cost['cost_per_1k'])} / 1k фото")}
    {metric_tile('Context / size', esc(str(market.get('context', 'n/a'))), esc(str(market.get('size', 'n/a'))))}
    {metric_tile('Latency p95', fmt_seconds(summary.get('p95_latency_sec')), f"avg {fmt_seconds(summary.get('avg_latency_sec'))}")}
    {metric_tile('Capacity', esc(capacity['serial_per_day']), f"{esc(capacity['workers_for_50k'])} workers for 50k/day")}
    {metric_tile('JSON/schema', fmt_pct_fraction(summary.get('schema_valid_rate')), f"parse {fmt_pct_fraction(summary.get('json_parse_rate'))}")}
    {metric_tile('Server reality', esc(fit_title(str(market.get('fit_level', 'unknown')))), esc(str(market.get('license', 'license n/a'))))}
  </div>
  <div class="model-note">
    <strong>Интерпретация:</strong> {esc(verdict)}
    <br><strong>Self-host:</strong> {esc(str(market.get('server_fit', 'нет данных')))}
    <br><strong>Прайс-источник:</strong> <a href="{esc(str(market.get('source', '#')))}">{esc(str(market.get('source', 'n/a')))}</a>
  </div>
  <div class="photo-stack">
    {photo_cards}
  </div>
</section>
"""


def metric_tile(label: str, value: str, detail: str) -> str:
    return f"<div class='metric-tile'><span>{label}</span><strong>{value}</strong><em>{detail}</em></div>"


def render_photo_card(row: dict[str, Any], images_dir: Path, output: Path) -> str:
    image = str(row["image"])
    parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
    expected = row.get("expected") if isinstance(row.get("expected"), dict) else {}
    image_path = (images_dir / image).resolve()
    image_src = os.path.relpath(image_path, output.parent.resolve()).replace(os.sep, "/")
    score = row_business_score(row)
    return f"""
<article class="photo-card">
  <figure>
    <img src="{esc(image_src)}" alt="{esc(image)}">
    <figcaption>{esc(image)}</figcaption>
  </figure>
  <div class="photo-body">
    <div class="photo-head">
      <div>
        <h3>{esc(image)}</h3>
        <p>{summary_line(parsed, expected)}</p>
      </div>
      <strong>{fmt_pct(score)}</strong>
    </div>
    {render_key_compare(parsed, expected)}
    {render_groups(row)}
  </div>
</article>
"""


def render_key_compare(parsed: dict[str, Any], expected: dict[str, Any]) -> str:
    fields = ["kik_present", "kik_sku_count", "kik_share_percent", "status_score"]
    rows = []
    for field in fields:
        rows.append(
            "<tr>"
            f"<th>{esc(FIELD_LABELS[field])}</th>"
            f"<td>{format_value(parsed.get(field), field)}</td>"
            f"<td>{format_value(expected.get(field), field)}</td>"
            f"<td>{delta_text(parsed.get(field), expected.get(field), field)}</td>"
            "</tr>"
        )
    return "<table class='key-compare'><thead><tr><th>Метрика</th><th>Ответ модели</th><th>GT</th><th>Δ</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def render_groups(row: dict[str, Any]) -> str:
    parts = []
    for title, fields in FIELD_GROUPS:
        cells = "".join(render_field_cell(row, field) for field in fields)
        parts.append(f"<details class='metric-group' open><summary>{esc(title)}</summary><div class='metric-grid'>{cells}</div></details>")
    return "\n".join(parts)


def render_field_cell(row: dict[str, Any], field: str) -> str:
    expected = row.get("expected") if isinstance(row.get("expected"), dict) else {}
    parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
    scores = row.get("field_scores") if isinstance(row.get("field_scores"), dict) else {}
    gt = expected.get(field)
    pred = parsed.get(field)
    state = field_state(gt, pred, scores.get(field), row.get("error"))
    return (
        f"<div class='field-cell {state['class']}'>"
        f"<div class='field-title'>{esc(FIELD_LABELS.get(field, field))}</div>"
        f"<div class='field-values'><span>model</span><b>{format_value(pred, field)}</b></div>"
        f"<div class='field-values'><span>GT</span><b>{format_value(gt, field)}</b></div>"
        f"<div class='field-status'>{esc(state['label'])}</div>"
        "</div>"
    )

def field_state(gt: Any, pred: Any, score: Any, error: Any) -> dict[str, str]:
    if error:
        return {"class": "bad", "label": "ошибка"}
    if gt is None or pred is None or score is None:
        return {"class": "nodata", "label": "нет данных"}
    value = as_float(score)
    if value is None:
        return {"class": "nodata", "label": "нет данных"}
    if value >= 0.999:
        return {"class": "ok", "label": "совпало"}
    if value > 0:
        return {"class": "partial", "label": f"частично {value:.2f}"}
    return {"class": "bad", "label": "не совпало"}


def row_business_score(row: dict[str, Any]) -> float | None:
    expected = row.get("expected") if isinstance(row.get("expected"), dict) else {}
    parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
    expected_sku_count = as_float(expected.get("kik_sku_count"))
    predicted_sku_count = as_float(parsed.get("kik_sku_count"))
    if expected_sku_count is not None and expected_sku_count > 0 and predicted_sku_count == 0:
        return 0.0
    scores = row.get("field_scores") if isinstance(row.get("field_scores"), dict) else {}
    numerator = 0.0
    denominator = 0.0
    for field, weight in BUSINESS_WEIGHTS.items():
        if field in scores:
            value = as_float(scores[field])
            if value is not None:
                numerator += value * weight
                denominator += weight
    if not denominator:
        return None
    return numerator / denominator * 100


def cost_stats(model_key: str, rows: list[dict[str, Any]]) -> dict[str, float | None]:
    market = MARKET_INFO.get(model_key, {})
    price_in = market.get("price_in")
    price_out = market.get("price_out")
    if price_in is None or price_out is None:
        return {"run_cost": None, "cost_per_1k": None}
    input_tokens = 0.0
    output_tokens = 0.0
    for row in rows:
        usage = row.get("token_usage") if isinstance(row.get("token_usage"), dict) else {}
        input_tokens += float(usage.get("input_tokens") or 0)
        output_tokens += float(usage.get("output_tokens") or 0)
    run_cost = input_tokens * float(price_in) / 1_000_000 + output_tokens * float(price_out) / 1_000_000
    per_1k = run_cost / len(rows) * 1000 if rows else None
    return {"run_cost": run_cost, "cost_per_1k": per_1k}


def capacity_stats(summary: dict[str, Any]) -> dict[str, str]:
    avg = as_float(summary.get("avg_latency_sec"))
    if not avg or avg <= 0:
        return {"serial_per_day": "n/a", "workers_for_50k": "n/a"}
    serial_per_day = 86400 / avg
    workers = max(1, math.ceil(50000 / serial_per_day))
    return {"serial_per_day": f"{serial_per_day:,.0f}/day".replace(",", " "), "workers_for_50k": str(workers)}


def model_verdict(model_key: str, summary: dict[str, Any]) -> str:
    if model_key == "qwen25_vl_72b":
        return "Лучший общий результат и самый разумный основной prod-кандидат, но critical_recall=0.5 запрещает hands-off automation без second-pass/manual review."
    if model_key == "mistral_large_3":
        return "Лучший heavy benchmark по KIK presence и SKU count, быстрый hosted-run, но для self-host это слишком тяжелый потолочный baseline."
    if model_key == "qwen3_vl_30b":
        return "Самый приятный по серверной реальности, однако в текущем prompt/schema плохо видит сам факт КИК и share; годится как дешевый кандидат после tuning."
    if model_key == "gemma4_31b":
        return "Хорошая execution-группа, но слабый core KIK и очень высокий p95 в этом прогоне; как primary кандидат пока не тянет."
    if model_key == "glm_46v":
        return "Лучший по share MAE и execution macro F1, но проседает SKU family и часто ошибается в наличии КИК; хороший fallback, не primary."
    if model_key == "qwen3_vl_235b":
        return "Несмотря на размер, провалился по core KIK; оставлять только как diagnostic benchmark, не внедрять."
    if model_key == "mistral_small_4":
        return "Самый быстрый hosted-run, но худший business score; reject для этой задачи без серьезной донастройки."
    return "Нет ручного вердикта."


def render_sources() -> str:
    links = []
    seen = set()
    for info in MARKET_INFO.values():
        source = info.get("source")
        if source and source not in seen:
            seen.add(source)
            links.append(f"<li><a href='{esc(str(source))}'>{esc(str(source))}</a></li>")
    return f"""
<section class="section sources">
  <div class="section-head">
    <p class="eyebrow">Источники market data</p>
    <h2>Прайс и context</h2>
  </div>
  <p>Прайсы и context length сверены по OpenRouter model pages на 2026-05-10; стоимость run считается из provider token_usage. Железо/self-host — инженерная оценка по размеру модели, не vendor SLA.</p>
  <ul>{''.join(links)}</ul>
</section>
"""


def summary_line(parsed: dict[str, Any], expected: dict[str, Any]) -> str:
    return (
        f"model: КИК {format_plain(parsed.get('kik_present'), 'kik_present')}, "
        f"{format_plain(parsed.get('kik_sku_count'), 'kik_sku_count')} SKU, "
        f"{format_plain(parsed.get('kik_share_percent'), 'kik_share_percent')} share; "
        f"GT: КИК {format_plain(expected.get('kik_present'), 'kik_present')}, "
        f"{format_plain(expected.get('kik_sku_count'), 'kik_sku_count')} SKU, "
        f"{format_plain(expected.get('kik_share_percent'), 'kik_share_percent')} share"
    )


def format_value(value: Any, field: str) -> str:
    return esc(format_plain(value, field))


def format_plain(value: Any, field: str) -> str:
    if value is None:
        return "нет данных"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if field == "status_score":
        return {0: "0 норма", 1: "1 внимание", 2: "2 критично"}.get(value, str(value))
    if field.endswith("_percent"):
        return f"{value}%"
    return str(value)


def delta_text(pred: Any, gt: Any, field: str) -> str:
    if pred is None or gt is None:
        return "n/a"
    if isinstance(pred, bool) or isinstance(gt, bool):
        return "ok" if pred == gt else "miss"
    pred_num = as_float(pred)
    gt_num = as_float(gt)
    if pred_num is None or gt_num is None:
        return "ok" if pred == gt else "miss"
    delta = pred_num - gt_num
    if delta == 0:
        return "0"
    suffix = " pp" if field.endswith("_percent") else ""
    return f"{delta:+.0f}{suffix}"


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_pct(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.1f}%"


def fmt_pct_fraction(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return "n/a"
    return f"{number * 100:.0f}%"


def fmt_num(value: Any, digits: int = 1) -> str:
    number = as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def fmt_seconds(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.1f}s"


def fmt_money(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return "n/a"
    if number < 0.01:
        return f"${number:.4f}"
    return f"${number:.2f}"


def fit_title(level: str) -> str:
    return {
        "good": "реально",
        "medium": "условно",
        "benchmark": "benchmark",
        "weak": "reject",
    }.get(level, "unknown")


def anchor(value: Any) -> str:
    return "model-" + str(value).replace("_", "-").replace("/", "-")


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def css() -> str:
    return """
:root {
  --ink: #171512;
  --muted: #66615a;
  --paper: #f6f2ea;
  --panel: #fffaf0;
  --line: #d7cec0;
  --green: #1f7a4d;
  --green-bg: #e5f5e9;
  --red: #aa2e27;
  --red-bg: #f9e1dd;
  --yellow: #906c00;
  --yellow-bg: #fff0bc;
  --gray: #747474;
  --gray-bg: #ededed;
  --blue: #214e7a;
  --blue-bg: #e1eef7;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  color: var(--ink);
  background: var(--paper);
  font-family: "Avenir Next", "Gill Sans", "Trebuchet MS", sans-serif;
  line-height: 1.45;
}
a { color: var(--blue); }
.mono, pre { font-family: "SFMono-Regular", "Cascadia Code", monospace; }
.top {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 280px;
  gap: 28px;
  padding: 44px clamp(18px, 5vw, 72px) 28px;
  border-bottom: 1px solid var(--line);
  background: linear-gradient(135deg, #fffdf7 0%, #f0eadf 58%, #e9f0ed 100%);
}
.eyebrow {
  margin: 0 0 10px;
  color: var(--blue);
  font-size: 12px;
  font-weight: 800;
  text-transform: uppercase;
}
h1, h2, h3 { letter-spacing: 0; line-height: 1.05; }
h1 { margin: 0; max-width: 1100px; font-family: Georgia, serif; font-size: clamp(36px, 6vw, 76px); }
h2 { margin: 0; font-family: Georgia, serif; font-size: clamp(28px, 4vw, 44px); }
h3 { margin: 0 0 8px; font-size: 20px; }
.lead { max-width: 980px; font-size: 18px; color: #3b3731; }
.hero-score {
  align-self: end;
  padding: 20px;
  border: 1px solid var(--ink);
  border-radius: 8px;
  background: #171512;
  color: #fffaf0;
}
.hero-score span, .hero-score em { display: block; color: #d9d1c6; font-style: normal; }
.hero-score strong { display: block; margin-top: 8px; font-size: 24px; }
.hero-score b { display: block; margin-top: 12px; font-family: Georgia, serif; font-size: 52px; }
.model-nav {
  position: sticky;
  top: 0;
  z-index: 5;
  display: flex;
  gap: 1px;
  overflow-x: auto;
  padding: 0 clamp(18px, 5vw, 72px);
  background: var(--ink);
}
.model-nav a {
  min-width: 164px;
  padding: 12px 14px;
  color: #fffaf0;
  text-decoration: none;
  border-right: 1px solid #38342f;
}
.model-nav span { display: block; color: #c8beb0; font-size: 12px; }
.model-nav b { display: block; margin-top: 4px; color: #ffe08a; }
main { padding: 26px clamp(18px, 5vw, 72px) 56px; }
.section {
  margin: 0 0 34px;
  padding: 28px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(255, 250, 240, 0.82);
}
.section-head { margin-bottom: 20px; }
.contract-grid, .rec-layout {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}
.plain-panel, .rec-aside {
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fffdf8;
}
ul { padding-left: 20px; }
.small, .footnote { color: var(--muted); font-size: 13px; }
.prompt-box {
  margin-top: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fffdf8;
}
.prompt-box summary { cursor: pointer; padding: 14px 16px; font-weight: 800; }
pre {
  white-space: pre-wrap;
  margin: 0;
  padding: 0 16px 16px;
  color: #302c28;
  font-size: 13px;
}
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }
table { width: 100%; border-collapse: collapse; background: #fffdf8; }
th, td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
th { color: #4b463f; font-size: 12px; text-transform: uppercase; }
td span { display: block; color: var(--muted); font-size: 12px; }
.rank-cell { font-weight: 900; }
.fit { display: block; max-width: 310px; }
.fit-good { color: var(--green); }
.fit-medium { color: var(--yellow); }
.fit-benchmark, .fit-weak { color: var(--red); }
.recommendation { background: #fdf8e8; border-color: #d2bb73; }
.rec-aside { background: var(--blue-bg); border-color: #a9c3d8; }
.model-section { padding: 0; overflow: hidden; }
.model-head {
  display: flex;
  justify-content: space-between;
  gap: 20px;
  padding: 26px 28px;
  background: #181613;
  color: #fffaf0;
}
.model-name { margin: 8px 0 0; color: #d4cabd; }
.big-score { align-self: center; font-family: Georgia, serif; font-size: 54px; color: #ffe08a; }
.model-dashboard {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  border-bottom: 1px solid var(--line);
  background: #fffdf8;
}
.metric-tile {
  min-height: 112px;
  padding: 16px;
  border-right: 1px solid var(--line);
}
.metric-tile span, .metric-tile em {
  display: block;
  color: var(--muted);
  font-size: 12px;
  font-style: normal;
}
.metric-tile strong {
  display: block;
  margin: 8px 0;
  font-size: 22px;
}
.model-note {
  padding: 18px 28px;
  background: #f8f4eb;
  border-bottom: 1px solid var(--line);
}
.photo-stack { padding: 18px; }
.photo-card {
  display: grid;
  grid-template-columns: minmax(260px, 34%) minmax(0, 1fr);
  gap: 18px;
  margin-bottom: 18px;
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fffdf8;
}
figure { margin: 0; }
figure img {
  width: 100%;
  max-height: 620px;
  object-fit: contain;
  border-radius: 6px;
  border: 1px solid var(--line);
  background: #efe7d8;
}
figcaption { margin-top: 8px; color: var(--muted); font-size: 13px; }
.photo-head {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  margin-bottom: 12px;
}
.photo-head p { margin: 0; color: var(--muted); }
.photo-head strong { font-size: 30px; color: var(--blue); white-space: nowrap; }
.key-compare { margin-bottom: 12px; font-size: 14px; }
.metric-group { margin-top: 10px; border: 1px solid var(--line); border-radius: 8px; background: #fffaf0; }
.metric-group summary { cursor: pointer; padding: 10px 12px; font-weight: 900; }
.metric-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
  padding: 0 10px 10px;
}
.field-cell {
  min-height: 118px;
  padding: 10px;
  border: 1px solid var(--line);
  border-left-width: 6px;
  border-radius: 8px;
  background: white;
}
.field-title { min-height: 34px; font-weight: 900; font-size: 13px; }
.field-values { display: grid; grid-template-columns: 44px minmax(0, 1fr); gap: 8px; font-size: 13px; }
.field-values span { color: var(--muted); }
.field-values b { overflow-wrap: anywhere; }
.field-status { margin-top: 8px; font-size: 12px; font-weight: 900; text-transform: uppercase; }
.ok { border-left-color: var(--green); background: var(--green-bg); }
.bad { border-left-color: var(--red); background: var(--red-bg); }
.partial { border-left-color: #d79b00; background: var(--yellow-bg); }
.nodata { border-left-color: var(--gray); background: var(--gray-bg); color: #555; }
.notes { margin-top: 12px; padding: 12px; border-radius: 8px; background: #f2f6f9; border: 1px solid #c6d8e5; }
.notes ul { margin: 6px 0 0; }
.sources ul { columns: 2; }
@media (max-width: 1100px) {
  .top, .contract-grid, .rec-layout, .photo-card { grid-template-columns: 1fr; }
  .model-dashboard { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 680px) {
  main { padding-inline: 12px; }
  .section { padding: 18px; }
  .model-head, .photo-head { display: block; }
  .model-dashboard, .metric-grid { grid-template-columns: 1fr; }
  .big-score { margin-top: 12px; }
  .photo-stack { padding: 10px; }
}
"""


if __name__ == "__main__":
    raise SystemExit(main())
