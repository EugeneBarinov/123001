#!/usr/bin/env python3
"""
Конфигурация для новостных источников
"""

import os
from typing import Dict, List

# Telegram конфигурация
TELEGRAM_CONFIG = {
    'api_id': os.environ.get('TELEGRAM_API_ID', ''),
    'api_hash': os.environ.get('TELEGRAM_API_HASH', ''),
    'phone': os.environ.get('TELEGRAM_PHONE', ''),
    'session_name': 'crypto_news_session'
}

# Каналы для мониторинга
TELEGRAM_CHANNELS = [
    "binanceupdates",      # Binance обновления
    "cryptocom",           # Crypto.com
    "coinbase",            # Coinbase
    "kraken",              # Kraken
    "bitfinex",            # Bitfinex
    "okx",                 # OKX
    "bybit",               # Bybit
    "kucoin",              # KuCoin
    "cryptonews",          # Общие криптоновости
    "bitcoinmagazine",     # Bitcoin Magazine
    "ethereum",            # Ethereum Foundation
    "cardano",             # Cardano
    "solana",              # Solana
    "polkadot",            # Polkadot
    "chainlink",           # Chainlink
    "uniswap",             # Uniswap
    "aave",                # Aave
    "compound",            # Compound
    "makerdao",            # MakerDAO
    "yearn",               # Yearn Finance
]

# TradingView конфигурация
TRADINGVIEW_CONFIG = {
    'base_url': 'https://www.tradingview.com',
    'news_endpoint': '/news/',
    'crypto_news_endpoint': '/news/cryptocurrency/',
    'headers': {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    },
    'request_delay': 1.0,  # Задержка между запросами в секундах
    'max_retries': 3
}

# Investing.com конфигурация
INVESTING_CONFIG = {
    'base_url': 'https://www.investing.com',
    'crypto_news_endpoint': '/cryptocurrency-news/',
    'headers': {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    },
    'request_delay': 1.5,
    'max_retries': 3
}

# Настройки сентимент-анализа
SENTIMENT_CONFIG = {
    'vader_threshold': 0.1,  # Порог для VADER
    'bert_model': 'ProsusAI/finbert',  # Модель BERT для финансов
    'max_text_length': 512,  # Максимальная длина текста для BERT
    'confidence_threshold': 0.6,  # Минимальная уверенность для классификации
}

# Категории новостей
NEWS_CATEGORIES = {
    'regulation': {
        'keywords': ['regulation', 'sec', 'cfdc', 'government', 'ban', 'legal', 'law', 'compliance'],
        'weight': 1.2  # Повышенный вес для регулятивных новостей
    },
    'adoption': {
        'keywords': ['adoption', 'partnership', 'enterprise', 'institutional', 'adoption', 'integration'],
        'weight': 1.1
    },
    'technology': {
        'keywords': ['upgrade', 'fork', 'protocol', 'smart contract', 'defi', 'nft', 'layer2', 'scaling'],
        'weight': 1.0
    },
    'market': {
        'keywords': ['bull', 'bear', 'rally', 'crash', 'volatility', 'liquidity', 'trading', 'volume'],
        'weight': 0.9
    },
    'mining': {
        'keywords': ['mining', 'hashrate', 'difficulty', 'block reward', 'halving'],
        'weight': 0.8
    },
    'exchange': {
        'keywords': ['exchange', 'listing', 'delisting', 'trading', 'volume', 'deposit', 'withdrawal'],
        'weight': 0.9
    },
    'defi': {
        'keywords': ['defi', 'yield', 'liquidity', 'amm', 'governance', 'dao', 'staking'],
        'weight': 1.0
    },
    'nft': {
        'keywords': ['nft', 'metaverse', 'gaming', 'art', 'collectibles', 'marketplace'],
        'weight': 0.8
    }
}

# Криптовалюты для отслеживания
CRYPTO_SYMBOLS = {
    'BTC': {
        'name': 'Bitcoin',
        'keywords': ['bitcoin', 'btc', 'btc/usd', 'bitcoin price'],
        'weight': 1.0
    },
    'ETH': {
        'name': 'Ethereum',
        'keywords': ['ethereum', 'eth', 'eth/usd', 'ethereum price'],
        'weight': 1.0
    },
    'BNB': {
        'name': 'Binance Coin',
        'keywords': ['binance coin', 'bnb', 'bnb/usd'],
        'weight': 0.9
    },
    'ADA': {
        'name': 'Cardano',
        'keywords': ['cardano', 'ada', 'ada/usd'],
        'weight': 0.8
    },
    'SOL': {
        'name': 'Solana',
        'keywords': ['solana', 'sol', 'sol/usd'],
        'weight': 0.8
    },
    'DOT': {
        'name': 'Polkadot',
        'keywords': ['polkadot', 'dot', 'dot/usd'],
        'weight': 0.8
    },
    'DOGE': {
        'name': 'Dogecoin',
        'keywords': ['dogecoin', 'doge', 'doge/usd'],
        'weight': 0.7
    },
    'MATIC': {
        'name': 'Polygon',
        'keywords': ['polygon', 'matic', 'matic/usd'],
        'weight': 0.8
    },
    'LTC': {
        'name': 'Litecoin',
        'keywords': ['litecoin', 'ltc', 'ltc/usd'],
        'weight': 0.7
    },
    'XRP': {
        'name': 'Ripple',
        'keywords': ['ripple', 'xrp', 'xrp/usd'],
        'weight': 0.8
    }
}

# Настройки базы данных
DATABASE_CONFIG = {
    'path': 'AI/data/enhanced_news.db',
    'max_news_items': 10000,  # Максимальное количество новостей в БД
    'cleanup_interval_hours': 24,  # Интервал очистки старых новостей
    'max_news_age_days': 30,  # Максимальный возраст новостей для хранения
}

# Настройки обновления
UPDATE_CONFIG = {
    'telegram_update_interval_minutes': 15,
    'tradingview_update_interval_minutes': 30,
    'investing_update_interval_minutes': 30,
    'sentiment_update_interval_minutes': 60,
    'max_concurrent_requests': 5,
    'request_timeout_seconds': 30,
}

# Настройки логирования
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'AI/logs/news_collector.log',
    'max_file_size_mb': 10,
    'backup_count': 5,
}

# Настройки фильтрации
FILTER_CONFIG = {
    'min_news_length': 20,  # Минимальная длина новости
    'max_news_length': 5000,  # Максимальная длина новости
    'exclude_keywords': ['spam', 'scam', 'fake', 'clickbait'],
    'min_sentiment_confidence': 0.3,  # Минимальная уверенность сентимента
    'min_impact_score': 0.1,  # Минимальный impact score
}

# Настройки экспорта
EXPORT_CONFIG = {
    'csv_encoding': 'utf-8',
    'json_indent': 2,
    'date_format': '%Y-%m-%d %H:%M:%S',
    'export_path': 'AI/data/exports/',
    'auto_export_interval_hours': 24,
}

def get_config() -> Dict:
    """Получаем полную конфигурацию"""
    return {
        'telegram': TELEGRAM_CONFIG,
        'tradingview': TRADINGVIEW_CONFIG,
        'investing': INVESTING_CONFIG,
        'sentiment': SENTIMENT_CONFIG,
        'categories': NEWS_CATEGORIES,
        'crypto_symbols': CRYPTO_SYMBOLS,
        'database': DATABASE_CONFIG,
        'update': UPDATE_CONFIG,
        'logging': LOGGING_CONFIG,
        'filter': FILTER_CONFIG,
        'export': EXPORT_CONFIG,
    }

def validate_config() -> bool:
    """Проверяем корректность конфигурации"""
    try:
        # Проверяем обязательные поля
        if not TELEGRAM_CONFIG['api_id'] or not TELEGRAM_CONFIG['api_hash']:
            print("⚠️ Telegram API не настроен")
        
        if not TELEGRAM_CHANNELS:
            print("⚠️ Не указаны каналы Telegram")
        
        # Проверяем пути
        os.makedirs(os.path.dirname(DATABASE_CONFIG['path']), exist_ok=True)
        os.makedirs(EXPORT_CONFIG['export_path'], exist_ok=True)
        os.makedirs(os.path.dirname(LOGGING_CONFIG['file']), exist_ok=True)
        
        print("✅ Конфигурация проверена")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка конфигурации: {e}")
        return False

if __name__ == "__main__":
    # Тестируем конфигурацию
    print("🔧 Тестирование конфигурации новостей...")
    
    if validate_config():
        config = get_config()
        print(f"📊 Категорий новостей: {len(config['categories'])}")
        print(f"🪙 Криптовалют: {len(config['crypto_symbols'])}")
        print(f"📺 Каналов Telegram: {len(TELEGRAM_CHANNELS)}")
        print("✅ Конфигурация готова к использованию")
    else:
        print("❌ Конфигурация требует исправления")
