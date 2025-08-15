# 📁 Структура проекта Crypto AI Analytics

## 🎯 Обзор
Проект реструктурирован для упрощения и лучшей организации. Убраны дублирующие файлы, создана четкая иерархия.

## 📂 Основная структура

```
AI/
├── 📁 api/                    # FastAPI сервер
│   ├── app.py                # Основной API сервер
│   └── __init__.py
├── 📁 config/                 # Конфигурация
│   ├── news_config.py        # Настройки новостей
│   └── __init__.py
├── 📁 data/                   # Данные и генерация
│   ├── generate_synthetic.py  # Синтетические данные
│   └── __init__.py
├── 📁 features/               # Технические индикаторы
│   ├── indicators.py          # Вычисление индикаторов
│   └── __init__.py
├── 📁 frontend/               # Пользовательский интерфейс
│   ├── enhanced_ui.py        # Основной UI (Streamlit)
│   └── __init__.py
├── 📁 inference/              # Инференс и предсказания
│   ├── predict_service.py     # Сервис предсказаний
│   ├── backtest.py           # Бэктестинг
│   └── __init__.py
├── 📁 models/                 # Модели машинного обучения
│   ├── enhanced_model.py     # Улучшенная нейросеть
│   ├── checkpoints/          # Сохраненные модели
│   └── __init__.py
├── 📁 news/                   # Анализ новостей
│   ├── enhanced_news.py      # Расширенный анализ
│   ├── store.py              # Хранение новостей
│   └── __init__.py
├── 📁 scripts/                # Вспомогательные скрипты
│   ├── check_imports.py      # Проверка импортов
│   └── __init__.py
├── 📁 github/                 # CI/CD
│   └── workflows/
├── 📁 logs/                   # Логи (создается автоматически)
├── 📁 assets/                 # Ресурсы (иконки, изображения)
├── 🚀 main.py                 # Главный файл приложения
├── 🔨 build_exe.py            # Сборка exe файла
├── 🚀 launcher.py             # Автоматический лаунчер
├── ⚙️ launcher_config.py      # Конфигурация лаунчера
├── 🖥️ start_app.bat          # Windows bat файл
├── 💻 start_app.ps1           # PowerShell скрипт
├── 📋 requirements.txt        # Основные зависимости
├── 📋 requirements_exe.txt    # Зависимости для exe
├── 🐳 Dockerfile              # Docker образ
├── 🐳 docker-compose.yml      # Docker Compose
└── 📚 README.md               # Основная документация
```

## 🎯 Ключевые файлы

### 🚀 Запуск приложения
- **`main.py`** - Главный файл для создания exe
- **`launcher.py`** - Автоматический запуск всех сервисов
- **`start_app.bat`** - Простой запуск для Windows
- **`start_app.ps1`** - Расширенный запуск для PowerShell

### 🔨 Создание exe
- **`build_exe.py`** - Автоматическая сборка exe файла
- **`requirements_exe.txt`** - Зависимости для сборки

### 🌐 Основные компоненты
- **`frontend/enhanced_ui.py`** - Современный UI интерфейс
- **`models/enhanced_model.py`** - Улучшенная нейросеть
- **`news/enhanced_news.py`** - Анализ новостей
- **`api/app.py`** - REST API сервер

## 🗑️ Удаленные файлы

Следующие файлы были удалены как дублирующие или устаревшие:
- ❌ `frontend/streamlit_app.py` → заменен на `enhanced_ui.py`
- ❌ `models/model.py` → заменен на `enhanced_model.py`
- ❌ `models/train.py` → функциональность в `enhanced_model.py`
- ❌ `models/sklearn_models.py` → интегрировано в `enhanced_model.py`
- ❌ `news/tg_ingest.py` → заменен на `enhanced_news.py`
- ❌ `news/sentiment.py` → интегрировано в `enhanced_news.py`
- ❌ `test_project.py` → функциональность в `launcher.py`
- ❌ `test_launcher.py` → функциональность в `build_exe.py`
- ❌ `PROJECT_STATUS.md` → информация в `README.md`

## 🚀 Способы запуска

### 1. Простой запуск (двойной клик)
```
AI/start_app.bat
```

### 2. PowerShell запуск
```powershell
cd AI
.\start_app.ps1
```

### 3. Python запуск
```bash
cd AI
python launcher.py
```

### 4. Главный файл
```bash
cd AI
python main.py
```

### 5. Exe файл (после сборки)
```
AI/dist/CryptoAI.exe
```

## 🔨 Сборка exe

### Автоматическая сборка
```bash
cd AI
python build_exe.py
```

### Ручная сборка с PyInstaller
```bash
cd AI
pip install -r requirements_exe.txt
pyinstaller --onefile --windowed --name=CryptoAI main.py
```

## 📊 Архитектура

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   🖥️ UI        │    │   🔌 API        │    │   🧠 Models     │
│  Streamlit      │◄──►│  FastAPI        │◄──►│  Neural Nets    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   📰 News      │    │   📊 Data       │    │   🔧 Features   │
│  Analysis       │    │  Generation     │    │  Indicators     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🎯 Преимущества новой структуры

✅ **Упрощено** - убраны дублирующие файлы  
✅ **Организовано** - четкая иерархия директорий  
✅ **Модульно** - каждый компонент в своем месте  
✅ **Готово к exe** - главный файл `main.py`  
✅ **Автозапуск** - лаунчер для всех сервисов  
✅ **Кроссплатформенно** - bat, ps1, python скрипты  

## 🔄 Миграция

Если у вас есть старые скрипты, обновите импорты:
- `from AI.frontend.streamlit_app import app` → `from AI.frontend.enhanced_ui import app`
- `from AI.models.model import LSTM` → `from AI.models.enhanced_model import CryptoAnalyzer`
- `from AI.news.tg_ingest import TelegramIngest` → `from AI.news.enhanced_news import NewsAnalyzer`
