# Отчет о проделанной работе по KIK VLM benchmark

Дата анализа: 2026-05-10  
Проект: `ice-cream-vlm-mvp`  
Задача: оценить VLM-модели на фото торгового оборудования с мороженым бренда «Коровка из Кореновки» / КИК.

## 1. Краткий вывод

Текущие результаты теста неудовлетворительны и не должны трактоваться как финальное доказательство плохого качества всех моделей.

Главное противоречие/проблема: ожидался прогон, где каждая модель на каждый target получает `6 JPG reference images + 1 target image + prompt`, но предыдущий сохраненный clean run `runs/kik_eval_7x10_merged/20260510_142125` был сделан без передачи reference images. Это прямо зафиксировано в README:

> `The latest cleaned benchmark run did not pass those reference images into the model request; prompt/reference wiring should be fixed before treating model quality as final.`

После этого была добавлена/проверена проводка референсов, и новые ref-runs действительно имеют `reference_count: 6`. Однако в новом 7x10 ref-прогоне часть результатов испорчена инфраструктурными и форматными сбоями:

- `gemma4_31b` почти полностью падала на OpenRouter `HTTP 429 temporarily rate-limited upstream`;
- `glm_46v` в первом ref-run часто не возвращала валидный JSON или обрезалась на `max_output_tokens=1024`;
- был сделан отдельный retry для `gemma4_31b` и `glm_46v`; после увеличения `glm_46v.max_output_tokens` до `2048` GLM восстановилась до `43.6554%`, но Gemma осталась нестабильной.

Да, по текущей архитектуре ref-run означает: для 10 target-картинок выполняется 10 отдельных вызовов на каждую модель; в каждом вызове передаются 6 reference JPG + 1 target image + prompt. Для 7 моделей это 70 model calls, и в каждом call 7 image inputs.

## 2. Что строилось

Был собран KIK-only VLM benchmark для retail execution photos:

- вход: фото торгового оборудования с мороженым;
- задача модели: вернуть strict JSON по бизнес-полям КИК;
- ground truth: ручная разметка `data/ground_truth/manual_ground_truth.jsonl`;
- датасет: 10 target-фото `data/real_images/photo_001.jpg` ... `photo_010.jpg`;
- reference set: 6 JPG в `data/reference_images/`:
  - `ref_cone.jpg`
  - `ref_cups.jpg`
  - `ref_eskimo.jpg`
  - `ref_lakomka.jpg`
  - `ref_large_formats.jpg`
  - `ref_sandwich.jpg`

## 3. Проверяемый I/O contract

Ожидаемый контракт одного model call:

1. System prompt: роль аудитора торгового оборудования КИК.
2. User prompt: список бизнес-полей, которые надо оценить.
3. Reference images: 6 изображений-справочников SKU-групп КИК.
4. Target image: одна оцениваемая фотография.
5. Output: один валидный JSON object, без markdown и текста вокруг.

Текущая реализация в `vlm_eval/providers.py` собирает multimodal payload так:

1. текст user prompt;
2. для каждого reference:
   - текст `REFERENCE IMAGE N: ... Use this only as a visual catalog reference... Do not score this image.`;
   - image_url с reference image;
3. текст `TARGET IMAGE TO ANALYZE... Return JSON for this target image only...`;
4. image_url с target image.

Важно: reference images идут перед target image, target явно подписан.

## 4. Модели

Конфиг: `vlm_eval/models.yaml`.

Тестировались 7 моделей:

- `qwen3_vl_235b`: Qwen/Qwen3-VL-235B-A22B-Instruct, quality ceiling, OpenRouter, heavy;
- `mistral_large_3`: mistralai/Mistral-Large-3-675B-Instruct-2512, quality ceiling, OpenRouter, heavy;
- `qwen3_vl_30b`: Qwen/Qwen3-VL-30B-A3B-Instruct, production candidate;
- `qwen25_vl_72b`: Qwen/Qwen2.5-VL-72B-Instruct, production candidate;
- `gemma4_31b`: google/gemma-4-31B-it, production candidate;
- `glm_46v`: zai-org/GLM-4.6V, production candidate;
- `mistral_small_4`: mistralai/Mistral-Small-4-119B-2603, production candidate.

Большинство моделей использовали `temperature=0`, `image_max_side=1024`, `response_format=json_schema`; `gemma4_31b` использовала `json_object`.

## 5. Схема ответа

Обязательные поля active runtime:

- тип фото/оборудования: `is_trade_equipment_photo`, `is_ice_cream_equipment`;
- кадрирование: `photo_crop_is_full`;
- ядро КИК: `kik_present`, `kik_sku_count`, `kik_share_percent`;
- SKU-группы: `has_cup`, `has_eskimo`, `has_lakomka`, `has_cone`, `has_sandwich`, `has_bucket`, `has_poleno_or_briquette`;
- выкладка/нарушения: `has_posm`, `has_monobrand_block`, `has_foreign_label`, `has_non_icecream_products`, `has_empty_sections`, `is_kik_mixed_with_competitors`;
- итог: `status_score`.

Схема strict: все поля обязательны, extra fields запрещены.

## 6. Scoring

Основная метрика: `kik_business_score_pct`.

Весовые группы:

- `kik_present`: 12;
- `kik_sku_count`: 12;
- `kik_share_percent`: 12;
- `has_poleno_or_briquette`: 3.5;
- SKU-группы суммарно значимо влияют на итог;
- `status_score`: 8;
- execution-поля: POSM, monobrand, mixed competitors, foreign labels, non-icecream, empty sections.

Для числовых полей используется bounded score:

- `kik_sku_count`: ошибка ограничивается cap=10;
- `kik_share_percent`: cap=50;
- score падает пропорционально ошибке.

Null в ground truth не скорится; parse/schema failures дают нули по scorable fields.

## 7. Прогоны

### 7.1 Старый merged clean run без референсов

Path: `runs/kik_eval_7x10_merged/20260510_142125`

Характеристики:

- 10 target images;
- 7 models;
- 70 rows in `results.jsonl`;
- 0 rows in `errors.jsonl`;
- config snapshot не содержит `reference_count`;
- README фиксирует, что этот run не передавал reference images в модель.

Рейтинг:

| model | score |
| --- | ---: |
| qwen25_vl_72b | 49.1644 |
| mistral_large_3 | 44.7969 |
| qwen3_vl_30b | 43.1786 |
| gemma4_31b | 41.4192 |
| glm_46v | 41.1354 |
| qwen3_vl_235b | 36.3971 |
| mistral_small_4 | 30.2171 |

Вывод: этот run полезен как исторический baseline без reference images, но не отвечает новому контракту `6 refs + target`.

### 7.2 Smoke/preflight ref runs

Paths:

- `runs/kik_eval_ref_smoke/20260510_214500`
- `runs/kik_eval_ref_preflight/20260510_214517`
- `runs/kik_eval_ref_smoke_after_strict/20260510_220435`

Назначение: проверить, что runner стартует и reference wiring работает на малом числе вызовов.

### 7.3 Основной ref-run 7 моделей x 10 фото

Path: `runs/kik_eval_ref_7x10/20260510_214604`

Характеристики из `config_snapshot.yaml`:

- `images: data/real_images`
- `limit: 10`
- `models: qwen3_vl_235b,mistral_large_3,qwen3_vl_30b,qwen25_vl_72b,gemma4_31b,glm_46v,mistral_small_4`
- `reference_count: 6`
- `references: data/reference_images/ref_cone.jpg,...,ref_sandwich.jpg`
- `concurrency: 3`
- `timeout_seconds: 90`

Файлы:

- `results.jsonl`: 70 rows;
- `errors.jsonl`: 18 rows.

Рейтинг по summary:

| model | score | schema_valid_rate | notes |
| --- | ---: | ---: | --- |
| qwen25_vl_72b | 63.1719 | 1.0 | лучший итоговый ref-run |
| qwen3_vl_30b | 58.9925 | 1.0 | второй |
| mistral_small_4 | 46.3730 | 1.0 | ниже production gate |
| mistral_large_3 | 41.6937 | 1.0 | quality ceiling сработал хуже ожидаемого |
| qwen3_vl_235b | 37.8414 | 1.0 | очень слабый core KIK |
| gemma4_31b | 6.5925 | 0.1 | результат испорчен 429 rate limits |
| glm_46v | 2.6728 | 0.1 | результат испорчен JSON/обрезанием |

Ключевая диагностика:

- `gemma4_31b`: 429 от upstream providers Parasail/Together через OpenRouter;
- `glm_46v`: ответы часто содержали chain-of-thought/explanatory text вместо JSON или обрывались на 1024 output tokens.

Вывод: этот run валиден по факту передачи референсов, но не полностью валиден как сравнение качества моделей из-за API/format failures у Gemma и GLM.

### 7.4 Retry для Gemma/GLM

Path: `runs/kik_eval_ref_retry_gemma_glm/20260510_220802`

Характеристики:

- models: `gemma4_31b,glm_46v`;
- limit: 10;
- reference_count: 6;
- concurrency: 1;
- `glm_46v.max_output_tokens`: 2048.

Результаты:

| model | score | schema_valid_rate | api/json errors |
| --- | ---: | ---: | ---: |
| glm_46v | 43.6554 | 1.0 | 0 |
| gemma4_31b | 20.1960 | 0.3 | 7 |

Вывод:

- GLM сильно лучше после увеличения token budget и более спокойного retry;
- Gemma осталась нестабильной/непригодной в текущем OpenRouter path из-за rate limits/schema failures.

### 7.5 Combined metric gate report

Path: `runs/kik_eval_ref_7x10_metric_gates/20260510_220802_combined`

Смысл: объединить основной ref-run с retry для `gemma4_31b` и `glm_46v`.

Источники:

- base run: `runs/kik_eval_ref_7x10/20260510_214604`;
- retry run: `runs/kik_eval_ref_retry_gemma_glm/20260510_220802`;
- retry models: `gemma4_31b`, `glm_46v`.

Combined ranking from `metric_gate_summary.csv`:

| model | business score | kik_present_f1 | sku MAE | share MAE | sku_family_f1 | schema_valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen25_vl_72b | 63.1719 | 0.8889 | 7.0 | 37.0 | 0.7605 | 1.0 |
| qwen3_vl_30b | 58.9925 | 0.8889 | 4.7 | 43.0 | 0.5271 | 1.0 |
| mistral_small_4 | 46.3730 | 0.8235 | 5.0 | 45.5 | 0.4997 | 1.0 |
| glm_46v | 43.6554 | 0.5714 | 7.5 | 39.0 | 0.5567 | 1.0 |
| mistral_large_3 | 41.6937 | 0.7500 | 5.5 | 52.0 | 0.6167 | 1.0 |
| qwen3_vl_235b | 37.8414 | 0.3333 | 8.0 | 55.5 | 0.3224 | 1.0 |
| gemma4_31b | 20.1960 | 0.4615 | 3.0 | 30.0 | 0.3405 | 0.3 |

Production gates в summary:

- 90-100%: excellent;
- 85-90%: strong production candidate;
- 75-85%: fallback/manual-assist;
- 60-75%: weak baseline;
- <60%: reject.

Hard minimum:

- `kik_present_f1 >= 0.95`
- `kik_sku_count_mae <= 1.5`
- `kik_share_percent_mae <= 10`
- `sku_family_macro_f1 >= 0.85`
- `critical_recall >= 0.90`
- `schema_valid_rate >= 0.98`

Ни одна модель не проходит hard minimum.

## 8. Что было изменено/добавлено в коде

Основные новые/измененные компоненты:

- `vlm_eval/run.py`: CLI runner, выбор моделей, загрузка cases, reference images, запуск model x image, retries, сохранение результатов;
- `vlm_eval/providers.py`: OpenAI-compatible provider, OpenRouter headers, multimodal payload, response_format fallback;
- `vlm_eval/models.yaml`: список моделей, provider model IDs, temperature, max tokens, image sizing, response format;
- `vlm_eval/tasks/kik/prompts.py`: system/user prompt и schema instruction;
- `vlm_eval/tasks/kik/schema.py`: strict JSON schema и validation;
- `vlm_eval/tasks/kik/scoring.py`: scoring, aggregation, business weights, metrics;
- `vlm_eval/tasks/kik/reporting.py`: запись summary/CSV/worst cases;
- `src/generate_kik_executive_report.py`: HTML executive report;
- `src/generate_kik_metric_gate_visualization.py`: визуализация metric gates;
- `tests/test_kik_eval.py`: unit tests.

Добавлены unit tests, включая проверку, что reference images отправляются перед target:

- `test_reference_images_are_sent_before_target`;
- `test_reference_image_discovery`;
- `test_kik_run_one_with_mock_provider`;
- schema/scoring/aggregation tests.

Последний запуск тестов:

- команда: `python3 -m unittest discover -s tests -v`
- результат: 13 tests OK.

## 9. Проверка вопроса: "получается прогон по 10 картинкам?"

Да, для одного выбранного model set это устроено так:

- есть 10 target images;
- для каждой target image создается отдельный model call;
- в каждом call передаются:
  - 6 reference JPG;
  - 1 target JPG;
  - prompt/schema instruction;
- для 7 моделей создается `7 * 10 = 70` строк результата.

То есть корректная формулировка:

> В ref-run каждая модель 10 раз вызывается на 10 target-фото. Каждый вызов получает 6 reference images + 1 target image + prompt. Итого на одну модель 10 calls, на 7 моделей 70 calls.

## 10. Почему результаты могут быть хуже, чем GPT-4.1o на первом запуске

Вероятные причины, в порядке важности:

1. Старый clean run не передавал reference images вообще, поэтому нельзя сравнивать его с ожидаемым ref-contract.
2. В ref-run добавление 6 изображений резко увеличило multimodal context и могло ухудшить внимание модели к target image.
3. Reference images могут быть восприняты не как справочник, а как часть сцены/объектов для анализа, несмотря на подписи.
4. Prompt требует слишком много полей за один проход: equipment type, SKU count, share, family detection, POSM, monobrand, foreign labels, empty sections, status.
5. SKU count и share percent требуют точного визуального подсчета по сложным витринам; это слабое место большинства VLM без специализированного детектора.
6. `max_output_tokens=512` для ряда моделей может быть маловат для strict JSON со всеми полями, особенно если модель начинает рассуждать или добавлять текст.
7. OpenRouter/provider behavior влияет на результат: rate limits, разные upstream providers, response_format incompatibility, JSON schema support.
8. Ground truth содержит сложные/частично субъективные поля; есть минимум один комментарий о противоречии POSM в разметке photo_005.
9. 10 изображений слишком мало для устойчивого ранжирования моделей; один-два плохих кейса сильно меняют итог.
10. Текущий score жестко штрафует parse/schema failures нулями, что правильно для production reliability, но смешивает "качество зрения" с "качество API/формата".

## 11. Самые важные несоответствия/риски

1. Не смешивать три разных типа результата:
   - no-reference baseline `kik_eval_7x10_merged`;
   - ref-run с failures `kik_eval_ref_7x10`;
   - combined ref metric gates после retry.
2. Нельзя называть Gemma результатом `6.59%` как чистое качество модели: это в основном OpenRouter/upstream 429 и schema failure.
3. Нельзя называть GLM результатом `2.67%` как чистое качество модели: после retry с 2048 tokens она стала `43.65%`.
4. Qwen2.5-VL-72B сейчас лучший в combined ref-run, но его `63.17%` все еще ниже production threshold и намного ниже ожидаемого уровня.
5. Quality-ceiling модели (`mistral_large_3`, `qwen3_vl_235b`) не показали ceiling behavior, что указывает либо на mismatch модели/задачи, либо на prompt/payload/scoring issues.

## 12. Что нужно проверить следующим шагом

Минимальный debug plan:

1. Сохранить raw request payload samples для 1-2 изображений и 2-3 моделей, чтобы GPT мог проверить порядок image blocks и wording.
2. Провести ablation:
   - target only;
   - target + 1 reference contact sheet;
   - target + 6 separate references;
   - target + текстовое описание SKU-групп без images.
3. Разделить задачу на 2-3 прохода:
   - pass A: KIK present + retail/ice-cream equipment check;
   - pass B: SKU groups/count/share;
   - pass C: execution/status.
4. Увеличить `max_output_tokens` минимум до 2048 для всех моделей со strict JSON.
5. Запретить reasoning более жестко и добавить короткий JSON skeleton/example.
6. Сделать reference contact sheet с крупными подписями, чтобы модель видела один reference image вместо 6 отдельных.
7. Пересмотреть scoring: отдельно считать API/schema reliability и business-field quality.
8. Проверить OpenRouter model IDs и provider support для `response_format=json_schema` по каждой модели.
9. Сделать контрольный GPT-4.1o/4.1 baseline на тех же 10 фото, с тем же exact payload contract.
10. Расширить golden set хотя бы до 30-50 фото перед финальным выводом.

## 13. Артефакты

Главные файлы:

- `README.md`
- `vlm_eval/run.py`
- `vlm_eval/providers.py`
- `vlm_eval/models.yaml`
- `vlm_eval/tasks/kik/prompts.py`
- `vlm_eval/tasks/kik/schema.py`
- `vlm_eval/tasks/kik/scoring.py`
- `tests/test_kik_eval.py`

Главные результаты:

- no-reference historical run: `runs/kik_eval_7x10_merged/20260510_142125`
- ref 7x10 run: `runs/kik_eval_ref_7x10/20260510_214604`
- Gemma/GLM retry: `runs/kik_eval_ref_retry_gemma_glm/20260510_220802`
- combined metric gates: `runs/kik_eval_ref_7x10_metric_gates/20260510_220802_combined`
- executive HTML: `runs/kik_eval_7x10_merged/20260510_142125/kik_executive_model_report.html`
- metric gate HTML: `runs/kik_eval_ref_7x10_metric_gates/20260510_220802_combined/kik_metric_gate_visualization.html`

## 14. Итоговая формулировка для внешнего анализа

Мы построили VLM benchmark для задачи retail execution по бренду КИК на 10 фото. Текущий правильный ref-contract: на каждую target-картинку модель получает 6 JPG reference images, затем target image и prompt/schema; для 10 фото это 10 calls на модель, для 7 моделей 70 calls. Старый clean run был без references, поэтому невалиден как проверка нового ref-contract. Новый ref-run с references показал лучший результат у Qwen2.5-VL-72B (`63.17%`), но все модели ниже production gates. Часть провалов вызвана не только model quality, а инфраструктурой и форматом: Gemma падала на OpenRouter 429, GLM при 1024 токенах не возвращала JSON/обрезалась; после retry с 2048 токенами GLM восстановилась до `43.65%`. Нужно разделить API/schema reliability и качество бизнес-полей, провести ablation по reference strategy и сравнить с GPT-4.1o на абсолютно идентичном payload.
