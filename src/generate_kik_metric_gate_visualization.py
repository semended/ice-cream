from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


BASE_RUN_DEFAULT = Path("runs/kik_eval_ref_7x10/20260510_214604")
RETRY_RUN_DEFAULT = Path("runs/kik_eval_ref_retry_gemma_glm/20260510_220802")
OUTPUT_DIR_DEFAULT = Path("runs/kik_eval_ref_7x10_metric_gates/20260510_220802_combined")
RETRY_MODELS_DEFAULT = ("gemma4_31b", "glm_46v")


GATES = [
    {
        "key": "schema_valid_rate",
        "label": "JSON/schema",
        "target": 0.98,
        "direction": "min",
        "format": "fraction_pct",
        "near": 0.90,
    },
    {
        "key": "kik_present_f1",
        "label": "KIK presence F1",
        "target": 0.95,
        "direction": "min",
        "format": "fraction",
        "near": 0.80,
    },
    {
        "key": "kik_sku_count_mae",
        "label": "SKU MAE",
        "target": 1.5,
        "direction": "max",
        "format": "number",
        "near": 3.0,
    },
    {
        "key": "kik_share_percent_mae",
        "label": "Share MAE",
        "target": 10.0,
        "direction": "max",
        "format": "pp",
        "near": 25.0,
    },
    {
        "key": "sku_family_macro_f1",
        "label": "SKU family F1",
        "target": 0.85,
        "direction": "min",
        "format": "fraction",
        "near": 0.70,
    },
    {
        "key": "critical_recall",
        "label": "Critical recall",
        "target": 0.90,
        "direction": "min",
        "format": "fraction",
        "near": 0.80,
    },
]


SCORE_BARS = [
    ("kik_business_score_pct", "Business score"),
    ("core_kik_score_pct", "Core KIK"),
    ("sku_family_score_pct", "SKU family"),
    ("execution_score_pct", "Execution"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate KIK metric gate visualization for model eval runs.")
    parser.add_argument("--base-run", type=Path, default=BASE_RUN_DEFAULT)
    parser.add_argument("--retry-run", type=Path, default=RETRY_RUN_DEFAULT)
    parser.add_argument("--retry-models", type=str, default=",".join(RETRY_MODELS_DEFAULT))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    retry_models = {item.strip() for item in args.retry_models.split(",") if item.strip()}
    rows = combine_summary(args.base_run, args.retry_run, retry_models)
    rows = sorted(rows, key=sort_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(args.output_dir / "metric_gate_summary.csv", rows)
    write_csv(args.output_dir / "metric_gate_cells.csv", gate_rows(rows))
    (args.output_dir / "merge_sources.json").write_text(
        json.dumps(
            {
                "base_run": args.base_run.as_posix(),
                "retry_run": args.retry_run.as_posix(),
                "retry_models": sorted(retry_models),
                "combined_models": [row["model_key"] for row in rows],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    output = args.output_dir / "kik_metric_gate_visualization.html"
    output.write_text(render_html(rows, args.base_run, args.retry_run, retry_models), encoding="utf-8")
    print(output)
    return 0


def combine_summary(base_run: Path, retry_run: Path, retry_models: set[str]) -> list[dict[str, Any]]:
    base_rows = {str(row["model_key"]): row for row in read_csv(base_run / "summary.csv")}
    if not retry_models:
        return list(base_rows.values())
    retry_rows = {str(row["model_key"]): row for row in read_csv(retry_run / "summary.csv")}
    for model_key in retry_models:
        if model_key in retry_rows:
            base_rows[model_key] = retry_rows[model_key]
    return list(base_rows.values())


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: coerce(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def coerce(value: str | None) -> Any:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def sort_key(row: dict[str, Any]) -> tuple[int, float]:
    failed = run_failed(row)
    return (1 if failed else 0, -(as_float(row.get("kik_business_score_pct")) or 0.0))


def gate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        for gate in GATES:
            state = gate_state(row, gate)
            output.append(
                {
                    "model_key": row["model_key"],
                    "metric": gate["key"],
                    "label": gate["label"],
                    "value": row.get(gate["key"]),
                    "target": gate["target"],
                    "status": state["status"],
                    "passes": state["status"] == "pass",
                    "run_failed": run_failed(row),
                }
            )
    return output


def render_html(rows: list[dict[str, Any]], base_run: Path, retry_run: Path, retry_models: set[str]) -> str:
    leader = rows[0]
    failed_models = [row for row in rows if run_failed(row)]
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang='ru'>",
            "<head>",
            "<meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1'>",
            "<title>KIK metric gates</title>",
            "<style>",
            css(),
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            render_hero(leader, failed_models, len(rows), base_run, retry_run, retry_models),
            render_score_board(rows),
            render_gate_heatmap(rows),
            render_detail_table(rows),
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def render_hero(
    leader: dict[str, Any],
    failed_models: list[dict[str, Any]],
    total_models: int,
    base_run: Path,
    retry_run: Path,
    retry_models: set[str],
) -> str:
    failed_text = ", ".join(str(row["model_key"]) for row in failed_models) or "нет"
    if retry_models:
        title = "Прохождение hard metrics после последнего retry"
        retry_text = ", ".join(f"<code>{esc(model)}</code>" for model in sorted(retry_models))
        source = (
            f"Base: <code>{esc(base_run.as_posix())}</code>. "
            f"Retry override: <code>{esc(retry_run.as_posix())}</code> для {retry_text}."
        )
    else:
        title = "Прохождение hard metrics по свежему run"
        source = (
            f"Run: <code>{esc(base_run.as_posix())}</code>. "
            "Без retry override: все значения взяты напрямую из <code>summary.csv</code>."
        )
    return f"""
<header class="hero">
  <p class="eyebrow">KIK VLM metric gates</p>
  <h1>{title}</h1>
  <div class="hero-grid">
    <section>
      <span>Лидер по business score</span>
      <strong>{esc(str(leader["model_key"]))}</strong>
      <b>{fmt_pct(leader.get("kik_business_score_pct"))}</b>
    </section>
    <section>
      <span>Отлетевшие по прогону</span>
      <strong>{esc(failed_text)}</strong>
      <b>{len(failed_models)} / {total_models}</b>
    </section>
    <section>
      <span>Hard gates</span>
      <strong>{len(GATES)} метрик</strong>
      <b>prod min</b>
    </section>
  </div>
  <p class="source">{source}</p>
</header>
"""


def render_score_board(rows: list[dict[str, Any]]) -> str:
    cards = []
    for row in rows:
        failed = run_failed(row)
        bars = []
        for key, label in SCORE_BARS:
            value = clamp(as_float(row.get(key)) or 0.0, 0.0, 100.0)
            bars.append(
                f"<div class='bar-row'><span>{esc(label)}</span><div class='bar'><i style='width:{value:.1f}%'></i></div><b>{value:.1f}%</b></div>"
            )
        cards.append(
            f"""
<article class="model-card {'is-failed' if failed else ''}">
  <div class="model-title">
    <h2>{esc(str(row['model_key']))}</h2>
    <span>{'run failed' if failed else 'run ok'}</span>
  </div>
  {''.join(bars)}
  <footer>{gate_pass_count(row)} / {len(GATES)} hard gates passed · errors {fmt_int(row.get('api_or_json_errors'))}/{fmt_int(row.get('total_cases'))}</footer>
</article>
"""
        )
    return "<section class='score-board'>" + "".join(cards) + "</section>"


def render_gate_heatmap(rows: list[dict[str, Any]]) -> str:
    header_cells = "".join(f"<th>{esc(str(gate['label']))}<small>{target_text(gate)}</small></th>" for gate in GATES)
    body = []
    for row in rows:
        cells = []
        for gate in GATES:
            state = gate_state(row, gate)
            cells.append(
                f"<td class='gate {state['status']}'><strong>{fmt_value(row.get(gate['key']), str(gate['format']))}</strong><span>{esc(state['label'])}</span></td>"
            )
        body.append(
            "<tr>"
            f"<th><span>{esc(str(row['model_key']))}</span><small>{gate_pass_count(row)}/{len(GATES)} pass</small></th>"
            + "".join(cells)
            + "</tr>"
        )
    return f"""
<section class="section">
  <div class="section-head">
    <p class="eyebrow">Hard minimum</p>
    <h2>Карта прохождения метрик</h2>
  </div>
  <div class="heatmap-wrap">
    <table class="heatmap">
      <thead><tr><th>Модель</th>{header_cells}</tr></thead>
      <tbody>{''.join(body)}</tbody>
    </table>
  </div>
</section>
"""


def render_detail_table(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{esc(str(row['model_key']))}</td>"
            f"<td>{fmt_pct(row.get('kik_business_score_pct'))}</td>"
            f"<td>{gate_pass_count(row)}/{len(GATES)}</td>"
            f"<td>{fmt_int(row.get('api_or_json_errors'))}/{fmt_int(row.get('total_cases'))}</td>"
            f"<td>{fmt_pct_fraction(row.get('schema_valid_rate'))}</td>"
            f"<td>{fmt_seconds(row.get('p95_latency_sec'))}</td>"
            f"<td>{model_note(row)}</td>"
            "</tr>"
        )
    return f"""
<section class="section">
  <div class="section-head">
    <p class="eyebrow">Short read</p>
    <h2>Итог по моделям</h2>
  </div>
  <div class="table-wrap">
    <table class="details">
      <thead><tr><th>Модель</th><th>Business</th><th>Gates</th><th>Errors</th><th>Schema</th><th>p95</th><th>Комментарий</th></tr></thead>
      <tbody>{''.join(body)}</tbody>
    </table>
  </div>
</section>
"""


def gate_state(row: dict[str, Any], gate: dict[str, Any]) -> dict[str, str]:
    value = as_float(row.get(gate["key"]))
    if value is None:
        return {"status": "missing", "label": "no data"}
    target = float(gate["target"])
    near = float(gate["near"])
    direction = str(gate["direction"])
    if direction == "min":
        if value >= target:
            return {"status": "pass", "label": "pass"}
        if value >= near:
            return {"status": "near", "label": "near"}
        return {"status": "fail", "label": "fail"}
    if value <= target:
        return {"status": "pass", "label": "pass"}
    if value <= near:
        return {"status": "near", "label": "near"}
    return {"status": "fail", "label": "fail"}


def gate_pass_count(row: dict[str, Any]) -> int:
    return sum(1 for gate in GATES if gate_state(row, gate)["status"] == "pass")


def run_failed(row: dict[str, Any]) -> bool:
    errors = as_float(row.get("api_or_json_errors")) or 0.0
    total = as_float(row.get("total_cases")) or 0.0
    schema = as_float(row.get("schema_valid_rate")) or 0.0
    return bool(total and errors / total > 0.5) or schema < 0.5


def model_note(row: dict[str, Any]) -> str:
    if run_failed(row):
        return "не засчитывать качество: прогон сломан API/JSON ошибками"
    passed = gate_pass_count(row)
    if passed == len(GATES):
        return "формально проходит hard minimum"
    if passed >= 2:
        return "есть полезный сигнал, но до prod threshold далеко"
    return "ниже hard minimum почти по всем ключевым метрикам"


def target_text(gate: dict[str, Any]) -> str:
    sign = ">=" if gate["direction"] == "min" else "<="
    return f"{sign} {fmt_value(gate['target'], str(gate['format']))}"


def fmt_value(value: Any, mode: str) -> str:
    number = as_float(value)
    if number is None:
        return "n/a"
    if mode == "fraction_pct":
        return f"{number * 100:.0f}%"
    if mode == "fraction":
        return f"{number:.2f}"
    if mode == "pp":
        return f"{number:.1f} pp"
    return f"{number:.1f}"


def fmt_pct(value: Any) -> str:
    number = as_float(value)
    return "n/a" if number is None else f"{number:.1f}%"


def fmt_pct_fraction(value: Any) -> str:
    number = as_float(value)
    return "n/a" if number is None else f"{number * 100:.0f}%"


def fmt_seconds(value: Any) -> str:
    number = as_float(value)
    return "n/a" if number is None else f"{number:.1f}s"


def fmt_int(value: Any) -> str:
    number = as_float(value)
    return "n/a" if number is None else str(int(number))


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if value != value:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def css() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f7f6f1;
  --ink: #1d2320;
  --muted: #66716b;
  --line: #d9d6cb;
  --panel: #ffffff;
  --green: #1f8a5b;
  --green-bg: #dff2e8;
  --yellow: #946300;
  --yellow-bg: #fff1c7;
  --red: #b13737;
  --red-bg: #f8dddd;
  --gray-bg: #ecebe6;
  --blue: #28688f;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main { width: min(1480px, calc(100% - 40px)); margin: 0 auto; padding: 32px 0 56px; }
.hero { padding: 28px 0 20px; border-bottom: 1px solid var(--line); }
.eyebrow { margin: 0 0 8px; color: var(--blue); text-transform: uppercase; font-weight: 800; letter-spacing: .08em; font-size: 12px; }
h1 { margin: 0; font-size: clamp(30px, 5vw, 58px); line-height: 1; letter-spacing: 0; }
h2 { margin: 0; font-size: 24px; letter-spacing: 0; }
.hero-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 24px; }
.hero-grid section, .model-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
}
.hero-grid span, .model-card footer, .bar-row span, .source, small { color: var(--muted); }
.hero-grid strong { display: block; margin-top: 6px; font-size: 24px; }
.hero-grid b { display: block; margin-top: 8px; font-size: 18px; color: var(--blue); }
.source { margin: 18px 0 0; }
code { background: #ecebe6; border-radius: 5px; padding: 2px 5px; }
.score-board { display: grid; grid-template-columns: repeat(7, minmax(190px, 1fr)); gap: 10px; margin: 24px 0; }
.model-card { min-height: 230px; }
.model-card.is-failed { border-color: #d7aaaa; background: #fff8f8; }
.model-title { display: flex; justify-content: space-between; gap: 10px; align-items: start; margin-bottom: 14px; }
.model-title h2 { font-size: 18px; overflow-wrap: anywhere; }
.model-title span { flex: 0 0 auto; border-radius: 999px; background: var(--green-bg); color: var(--green); padding: 3px 8px; font-size: 12px; font-weight: 800; }
.is-failed .model-title span { background: var(--red-bg); color: var(--red); }
.bar-row { display: grid; grid-template-columns: 88px 1fr 48px; gap: 8px; align-items: center; margin: 10px 0; }
.bar { height: 9px; border-radius: 999px; background: #e6e3da; overflow: hidden; }
.bar i { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #28688f, #1f8a5b); }
.section { margin-top: 34px; }
.section-head { display: flex; justify-content: space-between; gap: 20px; align-items: end; margin-bottom: 14px; }
.heatmap-wrap, .table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
thead th { background: #efede6; font-size: 12px; text-transform: uppercase; color: #46504b; }
thead small, tbody small { display: block; margin-top: 4px; text-transform: none; font-weight: 500; color: var(--muted); }
tbody tr:last-child th, tbody tr:last-child td { border-bottom: 0; }
.heatmap tbody th { min-width: 170px; }
.heatmap tbody th span { display: block; font-size: 16px; overflow-wrap: anywhere; }
.gate { min-width: 145px; border-left: 1px solid var(--line); }
.gate strong { display: block; font-size: 18px; }
.gate span { display: inline-block; margin-top: 6px; border-radius: 999px; padding: 3px 8px; font-size: 12px; font-weight: 800; }
.gate.pass { background: var(--green-bg); }
.gate.pass span { background: #fff; color: var(--green); }
.gate.near { background: var(--yellow-bg); }
.gate.near span { background: #fff; color: var(--yellow); }
.gate.fail { background: var(--red-bg); }
.gate.fail span { background: #fff; color: var(--red); }
.gate.missing { background: var(--gray-bg); }
.details td:first-child { font-weight: 800; }
@media (max-width: 1100px) {
  .score-board { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .hero-grid { grid-template-columns: 1fr; }
}
@media (max-width: 680px) {
  main { width: min(100% - 24px, 1480px); padding-top: 18px; }
  .score-board { grid-template-columns: 1fr; }
  .section-head { display: block; }
}
"""


if __name__ == "__main__":
    raise SystemExit(main())
