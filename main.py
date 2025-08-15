#!/usr/bin/env python3
"""
Crypto AI Analytics - Главное приложение
Основной файл для создания exe и запуска всего приложения
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

def start_api():
    """Запускает API сервер"""
    try:
        cmd = [sys.executable, "-m", "uvicorn", "api.app:app", "--host", "127.0.0.1", "--port", "8000"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return process
    except Exception as e:
        print(f"Ошибка запуска API: {e}")
        return None

def start_ui():
    """Запускает UI сервер"""
    try:
        cmd = [sys.executable, "-m", "streamlit", "run", "frontend/enhanced_ui.py", "--server.port", "8501", "--server.headless", "true"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return process
    except Exception as e:
        print(f"Ошибка запуска UI: {e}")
        return None

def main():
    """Главная функция приложения"""
    print("Запуск Crypto AI Analytics...")
    print("=" * 50)
    
    try:
        print("Конфигурация загружена")
        
        # Запускаем API
        print("Запуск API сервера...")
        api_process = start_api()
        if not api_process:
            print("Не удалось запустить API")
            return 1
        
        # Ждем запуска API
        time.sleep(3)
        
        # Запускаем UI
        print("Запуск UI сервера...")
        ui_process = start_ui()
        if not ui_process:
            print("Не удалось запустить UI")
            api_process.terminate()
            return 1
        
        # Ждем запуска UI
        time.sleep(5)
        
        print("Все сервисы запущены")
        print("Приложение успешно запущено!")
        print("UI доступен по адресу: http://localhost:8501")
        print("API доступен по адресу: http://127.0.0.1:8000")
        
        # Открываем браузер
        try:
            webbrowser.open("http://localhost:8501")
            print("Браузер открыт")
        except:
            print("Не удалось открыть браузер автоматически")
        
        print("\nДля остановки нажмите Ctrl+C")
        
        # Ждем завершения
        try:
            while True:
                time.sleep(1)
                # Проверяем, что процессы еще работают
                if api_process.poll() is not None:
                    print("API сервер остановлен")
                    break
                if ui_process.poll() is not None:
                    print("UI сервер остановлен")
                    break
        except KeyboardInterrupt:
            print("\nПолучен сигнал остановки...")
                
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        return 1
    finally:
        # Останавливаем все сервисы
        try:
            if 'api_process' in locals():
                api_process.terminate()
                print("API сервер остановлен")
            if 'ui_process' in locals():
                ui_process.terminate()
                print("UI сервер остановлен")
        except:
            pass
    
    print("Все сервисы остановлены")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
