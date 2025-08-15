#!/usr/bin/env python3
"""
Crypto AI Analytics Launcher
Автоматический запуск всех компонентов системы анализа криптовалют
"""

import os
import sys
import time
import subprocess
import threading
import signal
import webbrowser
from pathlib import Path
import logging
from typing import List, Optional

# Импорт конфигурации
try:
    from launcher_config import get_config, validate_config, get_api_url, get_ui_url
except ImportError:
    # Fallback конфигурация если модуль не найден
    def get_config():
        return {
            "api": {"host": "127.0.0.1", "port": 8000, "reload": True},
            "ui": {"host": "localhost", "port": 8501, "headless": True},
            "launcher": {"auto_open_browser": True, "startup_delay": 5},
            "monitoring": {"enabled": True, "check_interval": 30}
        }
    
    def validate_config(config): return True
    def get_api_url(config): return f"http://{config['api']['host']}:{config['api']['port']}"
    def get_ui_url(config): return f"http://{config['ui']['host']}:{config['ui']['port']}"

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('launcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CryptoAILauncher:
    """Главный класс для запуска всех компонентов системы"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.processes = []
        self.running = True
        
        # Загрузка конфигурации
        self.config = get_config()
        if not validate_config(self.config):
            raise ValueError("Невалидная конфигурация")
        
        # Сигналы для корректного завершения
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Обработчик сигналов для корректного завершения"""
        logger.info(f"Получен сигнал {signum}, завершаю работу...")
        self.running = False
        self.stop_all_services()
        sys.exit(0)
    
    def check_dependencies(self) -> bool:
        """Проверка наличия необходимых зависимостей"""
        logger.info("Проверка зависимостей...")
        
        try:
            import streamlit
            import uvicorn
            import torch
            import pandas
            import numpy
            import plotly
            logger.info("Все основные зависимости установлены")
            return True
        except ImportError as e:
            logger.error(f"Отсутствует зависимость: {e}")
            logger.info("Установите зависимости: pip install -r requirements.txt")
            return False
    
    def check_environment(self) -> bool:
        """Проверка окружения и конфигурации"""
        logger.info("Проверка окружения...")
        
        # Проверка виртуального окружения
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            logger.info("Виртуальное окружение активировано")
        else:
            logger.warning("Рекомендуется использовать виртуальное окружение")
        
        # Проверка файла .env для Telegram API
        env_file = self.project_root / "data" / ".env"
        if env_file.exists():
            logger.info("Файл .env найден")
        else:
            logger.warning("Файл .env не найден. Создайте его для работы с Telegram API")
        
        # Проверка моделей
        models_dir = self.project_root / "models" / "checkpoints"
        if models_dir.exists() and any(models_dir.iterdir()):
            logger.info("Модели найдены")
        else:
            logger.warning("Модели не найдены. Запустите обучение: python models/train.py")
        
        return True
    
    def start_api_service(self) -> bool:
        """Запуск API сервиса"""
        logger.info("Запуск API сервиса...")
        
        try:
            api_script = self.project_root / "api" / "app.py"
            if not api_script.exists():
                logger.error("API файл не найден")
                return False
            
            # Запуск API в отдельном процессе
            cmd = [
                sys.executable, "-m", "uvicorn", 
                "api.app:app", 
                "--host", self.config["api"]["host"], 
                "--port", str(self.config["api"]["port"])
            ]
            
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.processes.append(("API", process))
            logger.info(f"API сервис запущен на порту {self.config['api']['port']}")
            
            # Ждем запуска API
            time.sleep(3)
            return True
            
        except Exception as e:
            logger.error(f"Ошибка запуска API: {e}")
            return False
    
    def start_ui_service(self) -> bool:
        """Запуск UI сервиса"""
        logger.info("Запуск UI сервиса...")
        
        try:
            ui_script = self.project_root / "frontend" / "enhanced_ui.py"
            if not ui_script.exists():
                logger.error("UI файл не найден")
                return False
            
            # Запуск Streamlit в отдельном процессе
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                str(ui_script),
                "--server.port", str(self.config["ui"]["port"]),
                "--server.headless", "true" if self.config["ui"]["headless"] else "false"
            ]
            
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.processes.append(("UI", process))
            logger.info(f"UI сервис запущен на порту {self.config['ui']['port']}")
            
            # Ждем запуска UI
            time.sleep(5)
            return True
            
        except Exception as e:
            logger.error(f"Ошибка запуска UI: {e}")
            return False
    
    def check_service_health(self, service_name: str, url: str) -> bool:
        """Проверка здоровья сервиса"""
        try:
            import requests
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"{service_name} работает корректно")
                return True
            else:
                logger.warning(f"{service_name} вернул статус {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"{service_name} недоступен: {e}")
            return False
    
    def open_browser(self):
        """Открытие браузера с UI"""
        try:
            ui_url = f"http://localhost:{self.config['ui']['port']}"
            logger.info(f"Открываю браузер: {ui_url}")
            webbrowser.open(ui_url)
        except Exception as e:
            logger.error(f"Ошибка открытия браузера: {e}")
    
    def monitor_services(self):
        """Мониторинг работы сервисов"""
        logger.info("Запуск мониторинга сервисов...")
        
        while self.running:
            try:
                # Проверка API
                self.check_service_health("API", f"{get_api_url(self.config)}/health")
                
                # Проверка UI
                self.check_service_health("UI", get_ui_url(self.config))
                
                # Проверка процессов
                for name, process in self.processes:
                    if process.poll() is not None:
                        logger.error(f"{name} процесс завершился неожиданно")
                        self.running = False
                        break
                
                time.sleep(self.config["monitoring"]["check_interval"])
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Ошибка мониторинга: {e}")
                time.sleep(30)
    
    def stop_all_services(self):
        """Остановка всех сервисов"""
        logger.info("Остановка всех сервисов...")
        
        for name, process in self.processes:
            try:
                logger.info(f"Остановка {name}...")
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"Принудительная остановка {name}")
                process.kill()
            except Exception as e:
                logger.error(f"Ошибка остановки {name}: {e}")
        
        self.processes.clear()
        logger.info("Все сервисы остановлены")
    
    def start_all_services(self) -> bool:
        """Запуск всех сервисов"""
        logger.info("Запуск Crypto AI Analytics...")
        
        # Проверки
        if not self.check_dependencies():
            return False
        
        if not self.check_environment():
            logger.warning("Продолжаем запуск несмотря на предупреждения...")
        
        # Запуск сервисов
        if not self.start_api_service():
            logger.error("Не удалось запустить API")
            return False
        
        if not self.start_ui_service():
            logger.error("Не удалось запустить UI")
            self.stop_all_services()
            return False
        
        # Открытие браузера
        time.sleep(self.config["launcher"]["startup_delay"])
        if self.config["launcher"]["auto_open_browser"]:
            self.open_browser()
        
        # Запуск мониторинга в отдельном потоке
        monitor_thread = threading.Thread(target=self.monitor_services, daemon=True)
        monitor_thread.start()
        
        logger.info("Система успешно запущена!")
        logger.info(f"API: http://127.0.0.1:{self.config['api']['port']}")
        logger.info(f"UI: http://localhost:{self.config['ui']['port']}")
        logger.info("Нажмите Ctrl+C для остановки")
        
        return True
    
    def run(self):
        """Главный метод запуска"""
        if not self.start_all_services():
            return False
        
        try:
            # Основной цикл
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Получен сигнал остановки...")
        finally:
            self.stop_all_services()
        
        return True

def main():
    """Точка входа"""
    launcher = CryptoAILauncher()
    success = launcher.run()
    
    if success:
        logger.info("Система завершена корректно")
    else:
        logger.error("Система завершена с ошибками")
        sys.exit(1)

if __name__ == "__main__":
    main()
