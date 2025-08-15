#Requires -Version 5.1

<#
.SYNOPSIS
    🚀 Crypto AI Analytics Launcher
.DESCRIPTION
    Автоматический запуск всех компонентов системы анализа криптовалют
.PARAMETER NoBrowser
    Не открывать браузер автоматически
.PARAMETER Port
    Порт для API (по умолчанию 8000)
.EXAMPLE
    .\start_app.ps1
.EXAMPLE
    .\start_app.ps1 -NoBrowser -Port 8080
#>

param(
    [switch]$NoBrowser,
    [int]$Port = 8000
)

# Настройка кодировки для корректного отображения эмодзи
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "    🚀 Crypto AI Analytics Launcher" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Проверка Python
Write-Host "🔍 Проверка Python..." -ForegroundColor Blue
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $pythonVersion" -ForegroundColor Green
    } else {
        throw "Python не найден"
    }
} catch {
    Write-Host "❌ Python не найден в PATH" -ForegroundColor Red
    Write-Host "💡 Убедитесь, что Python установлен и добавлен в PATH" -ForegroundColor Yellow
    Write-Host "💡 Или активируйте виртуальное окружение" -ForegroundColor Yellow
    Read-Host "Нажмите Enter для выхода"
    exit 1
}

Write-Host ""

# Проверка виртуального окружения
Write-Host "🔍 Проверка виртуального окружения..." -ForegroundColor Blue
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "✅ Виртуальное окружение найдено" -ForegroundColor Green
    Write-Host "🔄 Активация виртуального окружения..." -ForegroundColor Blue
    
    try {
        & ".venv\Scripts\Activate.ps1"
        Write-Host "✅ Виртуальное окружение активировано" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Ошибка активации виртуального окружения" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️ Виртуальное окружение не найдено" -ForegroundColor Yellow
    Write-Host "💡 Рекомендуется создать виртуальное окружение" -ForegroundColor Yellow
}

Write-Host ""

# Проверка зависимостей
Write-Host "🔍 Проверка зависимостей..." -ForegroundColor Blue
try {
    $requirements = Get-Content "requirements.txt" -ErrorAction Stop
    Write-Host "✅ requirements.txt найден" -ForegroundColor Green
    
    # Проверка основных зависимостей
    $deps = @("streamlit", "uvicorn", "torch", "pandas", "numpy")
    foreach ($dep in $deps) {
        try {
            python -c "import $dep" 2>$null
            Write-Host "  ✅ $dep" -ForegroundColor Green
        } catch {
            Write-Host "  ❌ $dep" -ForegroundColor Red
        }
    }
} catch {
    Write-Host "⚠️ requirements.txt не найден" -ForegroundColor Yellow
}

Write-Host ""

# Запуск приложения
Write-Host "🚀 Запуск приложения..." -ForegroundColor Blue
Write-Host "💡 Это может занять несколько секунд..." -ForegroundColor Yellow
Write-Host ""

try {
    # Переход в директорию скрипта
    Set-Location $PSScriptRoot
    
    # Запуск лаунчера
    if ($NoBrowser) {
        $env:NO_BROWSER = "1"
    }
    
    if ($Port -ne 8000) {
        $env:API_PORT = $Port.ToString()
    }
    
    python launcher.py
    
} catch {
    Write-Host "❌ Ошибка запуска приложения: $_" -ForegroundColor Red
} finally {
    Write-Host ""
    Write-Host "🛑 Приложение остановлено" -ForegroundColor Yellow
    Read-Host "Нажмите Enter для выхода"
}
