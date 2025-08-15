#!/usr/bin/env python3
"""
🔍 Проверка структуры проекта Crypto AI Analytics
Проверяет наличие всех необходимых файлов и компонентов
"""

import os
from pathlib import Path
import importlib.util

def check_file_exists(file_path: str, description: str) -> bool:
    """Проверяет существование файла"""
    exists = Path(file_path).exists()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {file_path}")
    return exists

def check_directory_exists(dir_path: str, description: str) -> bool:
    """Проверяет существование директории"""
    exists = Path(dir_path).is_dir()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {dir_path}")
    return exists

def check_import(module_name: str, description: str) -> bool:
    """Проверяет возможность импорта модуля"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {module_name} - {e}")
        return False

def main():
    """Главная функция проверки"""
    print("🔍 Проверка структуры проекта Crypto AI Analytics")
    print("=" * 60)
    
    # Проверяем основные файлы
    print("\n📁 Основные файлы:")
    main_files = [
        ("main.py", "Главный файл приложения"),
        ("launcher.py", "Автоматический лаунчер"),
        ("launcher_config.py", "Конфигурация лаунчера"),
        ("build_exe.py", "Сборка exe файла"),
        ("start_app.bat", "Windows bat файл"),
        ("start_app.ps1", "PowerShell скрипт"),
        ("requirements.txt", "Основные зависимости"),
        ("requirements_exe.txt", "Зависимости для exe"),
        ("README.md", "Основная документация"),
        ("PROJECT_STRUCTURE.md", "Структура проекта")
    ]
    
    main_files_ok = 0
    for file_path, description in main_files:
        if check_file_exists(file_path, description):
            main_files_ok += 1
    
    # Проверяем основные директории
    print("\n📂 Основные директории:")
    main_dirs = [
        ("api", "API сервер"),
        ("config", "Конфигурация"),
        ("data", "Данные"),
        ("features", "Технические индикаторы"),
        ("frontend", "Пользовательский интерфейс"),
        ("inference", "Инференс и предсказания"),
        ("models", "Модели машинного обучения"),
        ("news", "Анализ новостей"),
        ("scripts", "Вспомогательные скрипты")
    ]
    
    main_dirs_ok = 0
    for dir_path, description in main_dirs:
        if check_directory_exists(dir_path, description):
            main_dirs_ok += 1
    
    # Проверяем ключевые компоненты
    print("\n🔧 Ключевые компоненты:")
    key_components = [
        ("frontend/enhanced_ui.py", "Современный UI"),
        ("models/enhanced_model.py", "Улучшенная нейросеть"),
        ("news/enhanced_news.py", "Анализ новостей"),
        ("api/app.py", "REST API сервер"),
        ("features/indicators.py", "Технические индикаторы"),
        ("inference/predict_service.py", "Сервис предсказаний")
    ]
    
    key_components_ok = 0
    for file_path, description in key_components:
        if check_file_exists(file_path, description):
            key_components_ok += 1
    
    # Проверяем импорты
    print("\n📦 Проверка импортов:")
    imports_ok = 0
    try:
        # Проверяем основные модули
        from launcher import CryptoAILauncher
        print("✅ Лаунчер: launcher.CryptoAILauncher")
        imports_ok += 1
    except ImportError as e:
        print(f"❌ Лаунчер: {e}")
    
    try:
        from launcher_config import get_config
        print("✅ Конфигурация: launcher_config.get_config")
        imports_ok += 1
    except ImportError as e:
        print(f"❌ Конфигурация: {e}")
    
    # Проверяем отсутствие удаленных файлов
    print("\n🗑️ Проверка удаленных файлов:")
    removed_files = [
        ("frontend/streamlit_app.py", "Старый UI (должен быть удален)"),
        ("models/model.py", "Старая модель (должна быть удалена)"),
        ("models/train.py", "Старый тренинг (должен быть удален)"),
        ("models/sklearn_models.py", "Старые sklearn модели (должны быть удалены)"),
        ("news/tg_ingest.py", "Старый Telegram инжект (должен быть удален)"),
        ("news/sentiment.py", "Старый анализ настроений (должен быть удален)")
    ]
    
    removed_files_ok = 0
    for file_path, description in removed_files:
        if not Path(file_path).exists():
            print(f"✅ {description}: {file_path} - удален")
            removed_files_ok += 1
        else:
            print(f"❌ {description}: {file_path} - все еще существует")
    
    # Итоговая статистика
    print("\n" + "=" * 60)
    print("📊 ИТОГОВАЯ СТАТИСТИКА:")
    print(f"📁 Основные файлы: {main_files_ok}/{len(main_files)} ✅")
    print(f"📂 Основные директории: {main_dirs_ok}/{len(main_dirs)} ✅")
    print(f"🔧 Ключевые компоненты: {key_components_ok}/{len(key_components)} ✅")
    print(f"📦 Импорты: {imports_ok}/2 ✅")
    print(f"🗑️ Удаленные файлы: {removed_files_ok}/{len(removed_files)} ✅")
    
    total_checks = len(main_files) + len(main_dirs) + len(key_components) + 2 + len(removed_files)
    total_passed = main_files_ok + main_dirs_ok + key_components_ok + imports_ok + removed_files_ok
    
    print(f"\n🎯 Общий результат: {total_passed}/{total_checks} проверок пройдено")
    
    if total_passed == total_checks:
        print("🎉 Проект полностью структурирован и готов к работе!")
        print("💡 Теперь можно создавать exe файл: python build_exe.py")
    else:
        print("⚠️ Есть проблемы, которые нужно исправить")
    
    return total_passed == total_checks

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Проверка не пройдена")
        exit(1)
    else:
        print("\n✅ Проверка пройдена успешно")
