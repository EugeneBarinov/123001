@echo off
chcp 65001 >nul
title Crypto AI Analytics Launcher

echo.
echo ========================================
echo    🚀 Crypto AI Analytics Launcher
echo ========================================
echo.

echo 🔍 Проверка Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python не найден в PATH
    echo 💡 Убедитесь, что Python установлен и добавлен в PATH
    echo 💡 Или активируйте виртуальное окружение
    pause
    exit /b 1
)

echo ✅ Python найден
echo.

echo 🔍 Проверка виртуального окружения...
if exist ".venv\Scripts\activate.bat" (
    echo ✅ Виртуальное окружение найдено
    echo 🔄 Активация виртуального окружения...
    call .venv\Scripts\activate.bat
    echo ✅ Виртуальное окружение активировано
) else (
    echo ⚠️ Виртуальное окружение не найдено
    echo 💡 Рекомендуется создать виртуальное окружение
)

echo.
echo 🚀 Запуск приложения...
echo 💡 Это может занять несколько секунд...
echo.

cd /d "%~dp0"
python launcher.py

echo.
echo 🛑 Приложение остановлено
pause
