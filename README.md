# ru-uz-mt — нейросетевой переводчик RU↔UZ (NLLB-200 + LoRA)

Система машинного перевода между русским и узбекским (оба направления: RU→UZ и UZ→RU) на базе
`facebook/nllb-200-distilled-600M`, дообученной методом LoRA/QLoRA. Проект покрывает полный
прикладной цикл: **данные → LoRA-дообучение → оценка → инференс → HTTP API в Docker**.

## Архитектура

```
 OPUS (wikimedia, TED2020, QED, Tatoeba, tldr-pages)
            │  scripts/download_data.sh
            ▼
 data/raw/<corpus>/*.ru|*.uz
            │  ru_uz_mt.data.prepare
            │  чистка → кириллица→латиница (uz) → фильтры длины/ratio/скрипта → дедуп
            │  сплит по ПАРАМ (без утечки) → раскрытие в оба направления
            ▼
 data/processed/{train,val,test}.jsonl   {src, tgt, direction, src_lang, tgt_lang}
            │  ru_uz_mt.train.train_lora
            │  NLLB-600M (опц. 4-bit NF4) + LoRA (peft) → Seq2SeqTrainer
            │  eval_loss + BLEU/chrF на val, early stopping
            ▼
 models/ru-uz-lora/   (только LoRA-адаптер, ~несколько МБ)
            │                                   │
            │  ru_uz_mt.eval.evaluate           │  ru_uz_mt.inference.Translator
            ▼                                   ▼
 results/metrics.json                    app/main.py (FastAPI) → Docker
 baseline vs fine-tuned                  /health, /translate, /translate/batch
```

| Модуль | Назначение |
|---|---|
| `src/ru_uz_mt/config.py` | Pydantic-конфиг, загрузка YAML + переопределение через env |
| `src/ru_uz_mt/data/normalize.py` | транслитерация узбекской кириллицы, нормализация апострофов, детект скрипта |
| `src/ru_uz_mt/data/prepare.py` | загрузка корпуса (OPUS / HF), чистка, сплит, JSONL |
| `src/ru_uz_mt/train/train_lora.py` | LoRA/QLoRA-дообучение, BLEU/chrF во время eval |
| `src/ru_uz_mt/inference/translator.py` | класс `Translator` (база + адаптер, `translate`, `translate_batch`) |
| `src/ru_uz_mt/eval/evaluate.py` | BLEU/chrF (+ опц. COMET), baseline vs fine-tuned |
| `app/main.py` | FastAPI-сервис |

## Почему NLLB и LoRA

- **NLLB-200 distilled 600M** — open-source MT-модель от Meta, официально поддерживает `rus_Cyrl` и
  `uzn_Latn`, помещается в одну GPU (и даже работает на CPU/Apple Silicon). Для чистого перевода
  специализированная MT-модель даёт лучшее качество на ватт, чем general-purpose LLM.
- **LoRA** обучает ~0.1–1% параметров: быстро, дёшево, адаптер весит единицы МБ, ниже риск
  catastrophic forgetting. **QLoRA** (4-bit NF4 через `bitsandbytes`) — опция для слабых GPU
  (`lora.use_qlora: true`, только CUDA).
- **Узбекский** — малоресурсный язык с двумя алфавитами. В препроцессинге узбекская сторона
  приводится к латинице (`uzn_Latn`): кириллица транслитерируется по официальной таблице 1995 г.
  (с контекстными правилами для `е`/`ц`) или отбрасывается (`normalize_uzbek_to_latin: false`);
  варианты апострофов (`o'`, `o‘`, `o’`) унифицируются в `oʻ` / `ʼ`.

## Данные

`Helsinki-NLP/opus-100` **не содержит пары ru-uz** (это англоцентричный корпус: только `en-xx`) —
проверено на этапе реализации. Поэтому по умолчанию используются корпуса OPUS напрямую
(Moses-формат, через OPUS API), актуальные на момент реализации:

| Корпус | Пар | Домен |
|---|---:|---|
| wikimedia | ~603k | Википедия (по умолчанию) |
| TED2020 | ~6k | выступления TED (по умолчанию) |
| QED | ~3k | образовательные субтитры (по умолчанию) |
| tldr-pages | ~0.7k | техническая документация (по умолчанию) |
| Tatoeba | ~0.2k | короткие фразы (по умолчанию) |
| Tanzil | ~85k | религиозные тексты (опционально) |
| GNOME, KDE4 | ~45k / ~37k | строки интерфейса ПО (опционально) |

Список корпусов — `data.opus_corpora` в конфиге и аргументы `scripts/download_data.sh`.
Можно использовать и любой HF-датасет с колонкой `translation: {ru, uz}`:
`data.source: hf`, `data.dataset_name: <name>`.

Фильтры: пустые строки, длина в словах (`min_len`/`max_len`), отношение длин (`length_ratio`),
доля нужного алфавита на каждой стороне, копии src==tgt, русский текст на узбекской стороне,
дедупликация, опц. fastText language-ID (`data.langid_model_path: lid.176.bin`, `pip install fasttext-wheel`).
Сплит делается по парам **до** раскрытия в два направления — одно и то же предложение не попадает
одновременно в train и test. Размер val/test ограничен `max_test_pairs` (по умолчанию 2000 пар).

## Установка

```bash
make install          # .venv + requirements-dev.txt + pip install -e .
cp .env.example .env  # при необходимости
```

Python 3.10+. Устройство выбирается автоматически: CUDA → Apple MPS → CPU
(`model.device` или `RU_UZ_MT__MODEL__DEVICE`).

## Запуск

```bash
make data    # скачать OPUS + подготовить data/processed/{train,val,test}.jsonl
make train   # LoRA-дообучение → models/ru-uz-lora/
make eval    # BLEU/chrF baseline vs fine-tuned → results/metrics.json
make serve   # FastAPI на :8000
```

Быстрый прогон на сабсете (демо/CI, работает и на ноутбуке):

```bash
make data  ARGS="--max-pairs 20000"
make train ARGS="--max-train-samples 4000 --max-eval-samples 200 --max-steps 200"
make eval  ARGS="--max-eval-samples 200"
```

Все гиперпараметры — в [`configs/train_config.yaml`](configs/train_config.yaml). Любое поле
переопределяется переменной окружения `RU_UZ_MT__<SECTION>__<FIELD>`, например
`RU_UZ_MT__TRAIN__NUM_EPOCHS=1 make train`. Логирование в MLflow/W&B — `train.report_to`.

## API

```bash
curl -s localhost:8000/health
# {"status":"ok","model_loaded":true,"device":"cuda","base_model":"facebook/nllb-200-distilled-600M","adapter":"models/ru-uz-lora"}

curl -s -X POST localhost:8000/translate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Привет, как дела?","src_lang":"rus_Cyrl","tgt_lang":"uzn_Latn"}'

curl -s -X POST localhost:8000/translate/batch \
  -H 'Content-Type: application/json' \
  -d '{"texts":["Salom!","Rahmat, yaxshi."],"src_lang":"uzn_Latn","tgt_lang":"rus_Cyrl"}'
```

Ограничения задаются env: `MAX_INPUT_CHARS` (2000), `MAX_BATCH_SIZE` (32), `REQUEST_TIMEOUT_S` (60).
Ошибки: 422 — невалидный вход/языки, 503 — модель не загружена, 504 — таймаут, 500 — ошибка генерации.
Swagger: `http://localhost:8000/docs`.

## Docker

```bash
make docker                       # GPU-образ (nvidia/cuda:12.1 runtime)
make docker VARIANT=cpu           # CPU-образ (python:3.10-slim)
docker compose up                 # API на :8000, models/ монтируется read-only
VARIANT=cpu docker compose up     # для CPU уберите блок deploy.resources в docker-compose.yml
```

GPU в контейнере требует NVIDIA Container Toolkit. Кэш Hugging Face хранится в volume `hf-cache`,
поэтому базовая модель скачивается один раз.

## Результаты

**Демо-прогон** (Apple M3 Pro, MPS): после чистки осталось 503k пар → ~1M направленных примеров.
LoRA r=16 на `q_proj, v_proj` (2.36M обучаемых параметров, 0.38%), **всего 150 шагов** × 16 примеров
(~2.4k примеров, ~3 мин), `max_seq_len=64`, lr 5e-4. Тест: первые 400 записей отложенного
test-сплита (204 RU→UZ, 196 UZ→RU), beam=4.

| direction | metric | baseline (NLLB-600M) | fine-tuned (LoRA) | Δ |
|---|---|---:|---:|---:|
| ru-uz | BLEU | 17.63 | 24.21 | +6.58 |
| ru-uz | chrF | 51.88 | 56.44 | +4.56 |
| uz-ru | BLEU | 28.90 | 28.78 | −0.12 |
| uz-ru | chrF | 53.38 | 54.35 | +0.97 |

Честная оговорка: часть прироста RU→UZ — это усвоенная орфография корпуса (`oʻ`, `gʻ`, `ʼ` вместо
ASCII `'`, которые выдаёт базовая NLLB). Если апострофы не учитывать, прирост RU→UZ составляет
**+2.0 chrF / +3.3 BLEU**. Тест-сет из того же распределения, что и train (в основном Википедия).
Полный прогон (`make train` с конфигом по умолчанию, 3 эпохи) ожидаемо даст больше; для него нужна CUDA GPU.

chrF — основная метрика: для агглютинативного узбекского она информативнее BLEU.

Команда демо-прогона:

```bash
RU_UZ_MT__TRAIN__BATCH_SIZE=4 RU_UZ_MT__TRAIN__GRAD_ACCUMULATION=4 \
RU_UZ_MT__TRAIN__MAX_SEQ_LEN=64 RU_UZ_MT__TRAIN__PAD_TO_MULTIPLE_OF=64 \
RU_UZ_MT__TRAIN__WARMUP_STEPS=20 RU_UZ_MT__TRAIN__EVAL_STEPS=150 RU_UZ_MT__TRAIN__SAVE_STEPS=150 \
RU_UZ_MT__TRAIN__LEARNING_RATE=5e-4 RU_UZ_MT__TRAIN__EVAL_BLEU_SAMPLES=40 \
make train ARGS="--max-train-samples 20000 --max-eval-samples 100 --max-steps 150"
make eval ARGS="--max-eval-samples 400"
```

> **Apple Silicon (MPS):** при динамическом паддинге аллокатор MPS фрагментируется: шаг
> деградирует с ~1.5 с до 100+ с, и система уходит в своп. Ставьте `pad_to_multiple_of` равным
> `max_seq_len` (фиксированные формы тензоров): тогда стабильно ~1.1 с/шаг.

## Тесты и линт

```bash
make test   # pytest: данные, Translator (моки, без скачивания модели), API (TestClient)
make lint   # ruff + black --check
```

## Ограничения и дальнейшие улучшения

- Корпус в основном из Википедии — доменный сдвиг для разговорной речи; тест-сет из того же
  распределения, стоит добавить внешний бенчмарк (FLORES-200 devtest, требует доступа к gated-датасету).
- Back-translation монолингвальных узбекских текстов для аугментации данных.
- Больше корпусов (NLLB-mined, доменные параллельные тексты), language-ID фильтр по умолчанию.
- LoRA на больше модулей (`k_proj`, `out_proj`, `fc1`, `fc2`) или NLLB-1.3B/3.3B.
- Гибрид с LLM: пост-редактирование / выбор лучшего из N гипотез, терминологические глоссарии.
- Сервинг: CTranslate2 / ONNX для ускорения, динамический батчинг запросов.

## Лицензия

MIT — см. [LICENSE](LICENSE). Данные OPUS и веса NLLB распространяются под собственными лицензиями
(NLLB-200 — CC-BY-NC 4.0).
