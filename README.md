# Ice Cream VLM MVP

MVP-контур для быстрой оценки VLM-моделей на задаче аудита полевых фото торгового оборудования с мороженым (бренд КИК/Ренна).

## Цель

Проект помогает сравнить VLM-кандидатов по качеству извлечения признаков из фото:
- тип оборудования;
- присутствие КИК;
- оценка SKU КИК;
- оценка доли КИК;
- оценка заполненности;
- видимые нарушения;
- рекомендованный статус точки.

## Данные

- `data/real_images/`: реальные полевые фото (target images) `photo_001.jpg ...`
- `data/ground_truth/kik_report_ground_truth.csv`: ручная разметка (MVP CSV)
- `data/raw/fair_prices.pdf`: продуктовая линейка с визуальными примерами (для reference images)

## Структура

```text
ice-cream-vlm-mvp/
  data/
    raw/
      fair_prices.pdf
    real_images/
    reference_images/
    reference_candidates/
    ground_truth/
      kik_report_ground_truth.csv
      kik_report_ground_truth_template.csv
      manual_ground_truth.jsonl
  results/
  src/
    extract_reference_images_from_pdf.py
    generate_ground_truth_jsonl.py
    run_openrouter_eval.py
    compare_with_ground_truth.py
  .env.example
  requirements.txt
  README.md
```

## Подготовка

1. Положите PDF в `data/raw/fair_prices.pdf`.
2. Создайте `.env` на основе примера:

```bash
cp .env.example .env
```

3. Вставьте в `.env` ваш ключ OpenRouter:

```env
OPENROUTER_API_KEY=...
```

## Запуск

```bash
pip install -r requirements.txt
python3 src/generate_ground_truth_jsonl.py
rm -f results/openrouter_eval_results.csv
python3 src/run_openrouter_eval.py
python3 src/compare_with_ground_truth.py
```

Безопасный малый прогон перед полным запуском:

```bash
OPENROUTER_MODELS="google/gemini-2.5-flash" MAX_IMAGES=2 python3 src/run_openrouter_eval.py
python3 src/compare_with_ground_truth.py
```

## Что делает каждый скрипт

- `src/extract_reference_images_from_pdf.py`:
  - извлекает embedded images из `data/raw/fair_prices.pdf` в `data/reference_candidates/embedded/`;
  - сохраняет `data/reference_candidates/reference_candidates_manifest.csv` (path/page/width/height/area);
  - рендерит страницы PDF и делает грубые crop-reference sheets по категориям в `data/reference_images/`;
  - сохраняет `data/reference_candidates/contact_sheet.jpg` для быстрой визуальной проверки.

- `src/generate_ground_truth_jsonl.py`:
  - читает `data/ground_truth/kik_report_ground_truth.csv` (поддерживает неполный набор колонок);
  - приводит значения к countable JSON-схеме (boolean/int/null);
  - автозаполняет часть бинарных полей из `expected_violations`, если явное поле не задано;
  - сохраняет `data/ground_truth/manual_ground_truth.jsonl` (1 строка = 1 image_id).

- `src/run_openrouter_eval.py`:
  - читает target images из `data/real_images/`;
  - если есть `data/ground_truth/manual_ground_truth.jsonl`, прогоняет только изображения, чьи имена есть в ground truth;
  - поддерживает `MAX_IMAGES=N` для малого прогона после ground truth-фильтрации;
  - поддерживает `OPENROUTER_MODELS="model_a,model_b"` для выбора моделей через env;
  - читает reference images из `data/reference_images/`;
  - отправляет в каждый запрос: prompt + reference images (с подписями) + target image;
  - прогоняет список моделей `MODELS` или список из `OPENROUTER_MODELS`;
  - просит JSON строго по JSON Schema (response_format json_schema);
  - сохраняет ответы/ошибки в `results/openrouter_eval_results.csv` (включая `prediction_json`).

- `src/compare_with_ground_truth.py`:
  - мержит предсказания с ground truth по `image_id`;
  - игнорирует поля, где ground truth = null;
  - считает MAE/RMSE (numeric) и accuracy/precision/recall/F1 (boolean);
  - считает coverage по каждому полю и модели;
  - сохраняет:
    - `results/model_comparison_details.csv`
    - `results/model_comparison_summary.csv`
    - `results/boolean_metrics_by_model.csv`
    - `results/numeric_metrics_by_model.csv`
    - `results/field_coverage_by_model.csv`
