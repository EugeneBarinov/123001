# 🚀 Crypto AI Analytics — Расширенная система анализа криптовалют

**Полноценная AI-система для анализа криптовалют с интеграцией новостей, технического анализа и машинного обучения**

## 🌟 Новые возможности

### 📰 **Расширенный анализ новостей**
- **Telegram интеграция** - мониторинг 20+ криптоканалов
- **TradingView парсинг** - автоматический сбор новостей
- **Investing.com интеграция** - финансовые новости в реальном времени
- **VADER + BERT сентимент-анализ** - точная оценка настроений
- **Классификация новостей** - 8 категорий с весами
- **Оценка влияния** - impact scoring для каждой новости

### 🤖 **Улучшенная нейросеть**
- **Transformer архитектура** - современный подход к анализу
- **Мультимодальный анализ** - цена + новости + индикаторы
- **Ensemble методы** - LSTM + GRU + Transformer
- **Attention механизмы** - фокус на важных паттернах
- **3-классная классификация** - UP/DOWN/NEUTRAL

### 📊 **Современный UI**
- **Интерактивные графики** - Plotly с candlestick
- **Технические индикаторы** - 26+ индикаторов
- **Анализ корреляций** - цена vs сентимент
- **Управление портфелем** - оптимизация аллокации
- **Риск-менеджмент** - калькуляторы позиций

## 🏗️ Архитектура системы

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Источники     │    │   Обработка     │    │     Анализ      │
│   новостей      │───▶│   и анализ      │───▶│   и ML          │
│                 │    │                 │    │                 │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Telegram      │    │ • Сентимент     │    │ • Transformer   │
│ • TradingView   │    │ • Классификация │    │ • Ensemble      │
│ • Investing.com │    │ • Impact Score  │    │ • Attention     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Хранение      │    │   API           │    │     UI          │
│   данных        │    │   (FastAPI)     │    │   (Streamlit)   │
│                 │    │                 │    │                 │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • SQLite        │    │ • REST API      │    │ • Дашборд       │
│ • CSV Export    │    │ • WebSocket     │    │ • Графики       │
│ • JSON API      │    │ • Real-time     │    │ • Управление    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Требования
- Python 3.10–3.12 (у вас уже есть venv в `.venv`)
- Windows PowerShell/Terminal
- (опционально) Docker Desktop (Compose v2: команда `docker compose`)

## 🔑 Настройка Telegram API

Для работы с новостями из Telegram необходимо:

1. **Получить API ключи:**
   - Перейдите на https://my.telegram.org/
   - Войдите с номером телефона
   - Создайте приложение и получите `api_id` и `api_hash`

2. **Создать файл `.env` в папке `AI/data/`:**
```bash
TELEGRAM_API_ID=your_api_id_here
TELEGRAM_API_HASH=your_api_hash_here
TELEGRAM_PHONE=+7XXXXXXXXXX
```

3. **Первый запуск:**
   - При первом запуске потребуется ввести код подтверждения
   - Код придет в Telegram
   - Сессия сохранится автоматически

## Быстрый старт (локально, Windows)
1) Активируйте окружение и установите зависимости:
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r AI\requirements.txt
```
2) Запустите API (в первом окне терминала):
```powershell
uvicorn AI.api.app:app --host 0.0.0.0 --port 8000
```
3) Запустите UI (во втором окне терминала):

**Базовый UI (старый):**
```powershell
streamlit run AI\frontend\streamlit_app.py
```

**Новый расширенный UI:**
```powershell
streamlit run AI\frontend\enhanced_ui.py
```

Если `streamlit` не найден:
```powershell
C:\VsCodeProject\.venv\Scripts\streamlit.exe run AI\frontend\enhanced_ui.py
```
4) Откройте браузер:
- UI: http://localhost:8501
- Проверка API: http://localhost:8000/health

### Панель процессов в UI
В левой боковой панели доступны кнопки:
- «Сгенерировать данные (синтетика)»: создаёт `AI/data/historical.csv`
- «Обучить LSTM (1 эпоха)»: обучает PyTorch-модель и сохраняет чекпоинт в `AI/models/checkpoints/model.pt`
- «Обучить sklearn (SVM/RF/LogReg)»: обучает и сохраняет модели в `AI/models/checkpoints/*.joblib`
- «Обновить новости (Telegram)»: заглушка, вставляет тестовые записи в `AI/data/news.db`
- Раздел «Логи процессов» показывает ход операций (автообновление)

В главной области:
- Поле пути к CSV (по умолчанию `AI/data/historical.csv`)
- Выбор модели: `pt` (LSTM), либо `svm`/`rf`/`logreg`
- Кнопки «Сделать предсказание» и «Запустить бэктест»
- График цены и блок логов

## Работа через API (при поднятом сервере)
- Здоровье:
```powershell
Invoke-WebRequest http://127.0.0.1:8000/health
```
- Логи (последние строки):
```powershell
Invoke-WebRequest http://127.0.0.1:8000/logs
```
- Генерация синтетики:
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/data/generate
```
- Обучение LSTM (1 эпоха):
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/train/pt -ContentType application/json -Body '{}'
```
- Обучение sklearn:
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/train/sklearn -ContentType application/json -Body '{}'
```
- Предсказание (модель по выбору, например SVM):
```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict?model=svm" -ContentType application/json -Body '{"csv_path":"AI/data/historical.csv"}'
```
- Бэктест (модель по выбору):
```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/backtest?model=pt" -ContentType application/json -Body '{"csv_path":"AI/data/historical.csv"}'
```
- Обновить новости (заглушка):
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/news/refresh -ContentType application/json -Body '{}'
```

## CLI-скрипты (альтернатива UI)
- Сгенерировать синтетические данные:
```powershell
python -m AI.data.generate_synthetic --rows 1500 --out AI/data/historical.csv
```
- Обучить LSTM:
```powershell
python -m AI.models.train --csv AI/data/historical.csv --epochs 3 --batch 64 --seq_len 60
```
- Обучить sklearn-модели:
```powershell
python -m AI.models.sklearn_models --csv AI/data/historical.csv --out_dir AI/models/checkpoints
```

## Переменные окружения
- MODEL_CKPT: путь к чекпоинту LSTM (по умолчанию `AI/models/checkpoints/model.pt`)
- MODEL_SVM / MODEL_RF / MODEL_LOGREG: пути к joblib-моделям
- UI использует `API_URL` (по умолчанию `http://localhost:8000`)

## Docker
Требуется Docker Desktop с Compose v2.

- Сборка API-образа вручную:
```powershell
cd AI
docker build -t crypto-ai .
```
- Запуск API (с монтированием проекта внутрь):
```powershell
docker run --rm -p 8000:8000 -v "${PWD}:/app" -e MODEL_CKPT=AI/models/checkpoints/model.pt crypto-ai
```
- Запуск API+UI через Compose v2:
```powershell
cd AI
docker compose up --build
```
Открыть UI: http://localhost:8501

Примечание: если команда `docker compose` не найдена — установите/обновите Docker Desktop. В Compose v2 используется пробел (не `docker-compose`).

## Эндпоинты API
- GET `/health` — статус
- GET `/logs` — последние логи операций
- POST `/data/generate` — создать синтетические данные
- POST `/train/pt` — обучить LSTM (параметры: `csv_path`, `epochs`)
- POST `/train/sklearn` — обучить sklearn-модели
- POST `/predict?model=pt|svm|rf|logreg` — предсказание по CSV
- POST `/backtest?model=pt|svm|rf|logreg` — бэктест
- POST `/news/refresh` — обновление новостей (Telegram заглушка)

## 🚀 Автоматический запуск

### Новый лаунчер-приложение

Для автоматического запуска всех компонентов без ручных команд используйте новый лаунчер:

**Windows (двойной клик):**
```
AI\start_app.bat
```

**PowerShell:**
```powershell
cd AI
.\start_app.ps1
```

**Python:**
```bash
cd AI
python launcher.py
```

**Главный файл:**
```bash
cd AI
python main.py
```

Лаунчер автоматически:
- ✅ Запустит API сервис на порту 8000
- ✅ Запустит UI интерфейс на порту 8501
- ✅ Откроет браузер с приложением
- ✅ Будет мониторить здоровье сервисов
- ✅ Корректно завершит все процессы при остановке

📚 **Подробная документация:** [LAUNCHER_README.md](LAUNCHER_README.md)

## 🔨 Создание exe файла

### Автоматическая сборка
```bash
cd AI
python build_exe.py
```

### Ручная сборка
```bash
cd AI
pip install -r requirements_exe.txt
pyinstaller --onefile --windowed --name=CryptoAI main.py
```

После сборки exe файл будет доступен в `AI/dist/CryptoAI.exe`

## 📁 Структура проекта

Проект реструктурирован для упрощения и лучшей организации:

- 🚀 **`main.py`** - Главный файл приложения для exe
- 🔨 **`build_exe.py`** - Автоматическая сборка exe
- 🚀 **`launcher.py`** - Автоматический запуск всех сервисов
- 📊 **`frontend/enhanced_ui.py`** - Современный UI интерфейс
- 🧠 **`models/enhanced_model.py`** - Улучшенная нейросеть
- 📰 **`news/enhanced_news.py`** - Анализ новостей

📚 **Подробная структура:** [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## Примечания
- Файл `AI/data/historical.csv` при первом запуске может быть пустым (только заголовок). Сгенерируйте данные или загрузите реальные.
- Для Binance реальных данных настройте ключи в `AI/data/.env` (см. `credentials_example.env`) и используйте `AI/data/download_historical.py`.
- Новости: сейчас заглушка. Для реального Telethon понадобятся API ID/Hash и список каналов.
