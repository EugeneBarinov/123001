"""
Конфигурация лаунчера Crypto AI Analytics
Настройки для автоматического запуска всех компонентов
"""

import os
from typing import Dict, Any

# Основные настройки
DEFAULT_CONFIG = {
    # Порты сервисов
    "api": {
        "host": "127.0.0.1",
        "port": 8000,
        "reload": True,
        "workers": 1
    },
    
    "ui": {
        "host": "localhost",
        "port": 8501,
        "headless": True,
        "theme": "dark"
    },
    
    # Настройки запуска
    "launcher": {
        "auto_open_browser": True,
        "check_health_interval": 30,  # секунды
        "startup_delay": 5,  # секунды
        "log_level": "INFO"
    },
    
    # Настройки мониторинга
    "monitoring": {
        "enabled": True,
        "check_interval": 30,  # секунды
        "max_retries": 3,
        "health_timeout": 5  # секунды
    },
    
    # Настройки логирования
    "logging": {
        "file": "launcher.log",
        "max_size": "10MB",
        "backup_count": 5,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    },
    
    # Пути к файлам
    "paths": {
        "api_script": "api/app.py",
        "ui_script": "frontend/enhanced_ui.py",
        "models_dir": "models/checkpoints",
        "data_dir": "data",
        "logs_dir": "logs"
    },
    
    # Настройки зависимостей
    "dependencies": {
        "required": [
            "streamlit",
            "uvicorn",
            "torch",
            "pandas",
            "numpy",
            "plotly"
        ],
        "optional": [
            "transformers",
            "telethon",
            "vaderSentiment",
            "beautifulsoup4",
            "textblob"
        ]
    }
}

def get_config() -> Dict[str, Any]:
    """Получение конфигурации с учетом переменных окружения"""
    config = DEFAULT_CONFIG.copy()
    
    # Переопределение из переменных окружения
    env_mappings = {
        "API_HOST": ("api", "host"),
        "API_PORT": ("api", "port"),
        "UI_PORT": ("ui", "port"),
        "NO_BROWSER": ("launcher", "auto_open_browser"),
        "LOG_LEVEL": ("launcher", "log_level"),
        "HEALTH_INTERVAL": ("monitoring", "check_interval"),
        "STARTUP_DELAY": ("launcher", "startup_delay")
    }
    
    for env_var, config_path in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            section, key = config_path
            if key == "port":
                try:
                    config[section][key] = int(env_value)
                except ValueError:
                    pass
            elif key == "auto_open_browser":
                config[section][key] = env_value.lower() not in ("0", "false", "no")
            elif key == "check_interval":
                try:
                    config[section][key] = int(env_value)
                except ValueError:
                    pass
            elif key == "startup_delay":
                try:
                    config[section][key] = int(env_value)
                except ValueError:
                    pass
            else:
                config[section][key] = env_value
    
    return config

def validate_config(config: Dict[str, Any]) -> bool:
    """Валидация конфигурации"""
    try:
        # Проверка портов
        if not (1 <= config["api"]["port"] <= 65535):
            return False
        if not (1 <= config["ui"]["port"] <= 65535):
            return False
        
        # Проверка интервалов
        if config["monitoring"]["check_interval"] < 1:
            return False
        if config["launcher"]["startup_delay"] < 0:
            return False
        
        return True
    except (KeyError, TypeError):
        return False

def get_api_url(config: Dict[str, Any]) -> str:
    """Получение URL для API"""
    return f"http://{config['api']['host']}:{config['api']['port']}"

def get_ui_url(config: Dict[str, Any]) -> str:
    """Получение URL для UI"""
    return f"http://{config['ui']['host']}:{config['ui']['port']}"

def print_config(config: Dict[str, Any]):
    """Вывод конфигурации в консоль"""
    print("Конфигурация лаунчера:")
    print(f"  API: {get_api_url(config)}")
    print(f"  UI: {get_ui_url(config)}")
    print(f"  Автозапуск браузера: {'Да' if config['launcher']['auto_open_browser'] else 'Нет'}")
    print(f"  Мониторинг: {'Включен' if config['monitoring']['enabled'] else 'Отключен'}")
    print(f"  Лог-уровень: {config['launcher']['log_level']}")

if __name__ == "__main__":
    # Тестирование конфигурации
    config = get_config()
    print_config(config)
    
    if validate_config(config):
        print("Конфигурация валидна")
    else:
        print("Конфигурация невалидна")
