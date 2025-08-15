#!/usr/bin/env python3
"""
Расширенный новостной модуль для анализа криптовалют
- Telegram каналы
- TradingView
- Investing.com
- Сентимент-анализ (VADER + BERT)
- Классификация новостей
"""

import asyncio
import re
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NewsItem:
    """Расширенная структура новости"""
    id: str
    source: str
    channel: str
    title: str
    text: str
    url: str
    date_utc: datetime
    sentiment_score: float
    sentiment_label: str
    category: str
    impact_score: float
    crypto_mentions: List[str]
    price_impact: Optional[float] = None

@dataclass
class NewsSource:
    """Конфигурация источника новостей"""
    name: str
    type: str  # telegram, tradingview, investing
    url: str
    api_key: Optional[str] = None
    channels: List[str] = None

class SentimentAnalyzer:
    """Продвинутый анализатор сентимента"""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.bert_model = None
        self.bert_tokenizer = None
        self._load_bert()
    
    def _load_bert(self):
        """Загружаем BERT модель для более точного анализа"""
        try:
            # Используем финансовую модель для лучшего анализа
            model_name = "ProsusAI/finbert"
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            logger.info("BERT модель загружена успешно")
        except Exception as e:
            logger.warning(f"BERT модель не загружена: {e}")
            self.bert_model = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Анализируем сентимент текста"""
        # VADER анализ
        vader_scores = self.vader.polarity_scores(text)
        
        # BERT анализ (если доступен)
        bert_score = None
        if self.bert_model and len(text) > 10:
            try:
                inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)
                    bert_score = float(probs[0][1])  # Positive probability
            except Exception as e:
                logger.warning(f"BERT анализ не удался: {e}")
        
        # Комбинированный анализ
        if bert_score is not None:
            # Взвешенное среднее VADER и BERT
            combined_score = 0.4 * vader_scores['compound'] + 0.6 * (bert_score - 0.5) * 2
        else:
            combined_score = vader_scores['compound']
        
        # Определяем лейбл
        if combined_score >= 0.1:
            label = "POSITIVE"
        elif combined_score <= -0.1:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        
        return {
            "vader_compound": vader_scores['compound'],
            "vader_positive": vader_scores['pos'],
            "vader_negative": vader_scores['neg'],
            "vader_neutral": vader_scores['neu'],
            "bert_score": bert_score,
            "combined_score": combined_score,
            "label": label
        }

class NewsClassifier:
    """Классификатор новостей по категориям"""
    
    def __init__(self):
        self.categories = {
            "regulation": ["regulation", "sec", "cfdc", "government", "ban", "legal"],
            "adoption": ["adoption", "partnership", "enterprise", "institutional", "adoption"],
            "technology": ["upgrade", "fork", "protocol", "smart contract", "defi", "nft"],
            "market": ["bull", "bear", "rally", "crash", "volatility", "liquidity"],
            "mining": ["mining", "hashrate", "difficulty", "block reward"],
            "exchange": ["exchange", "listing", "delisting", "trading", "volume"]
        }
    
    def classify_news(self, text: str) -> str:
        """Классифицируем новость по категории"""
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in self.categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[category] = score
        
        if not any(scores.values()):
            return "general"
        
        return max(scores, key=scores.get)

class CryptoMentionExtractor:
    """Извлекаем упоминания криптовалют"""
    
    def __init__(self):
        # Основные криптовалюты
        self.crypto_symbols = {
            "BTC": ["bitcoin", "btc"],
            "ETH": ["ethereum", "eth"],
            "BNB": ["binance coin", "bnb"],
            "ADA": ["cardano", "ada"],
            "SOL": ["solana", "sol"],
            "DOT": ["polkadot", "dot"],
            "DOGE": ["dogecoin", "doge"],
            "MATIC": ["polygon", "matic"],
            "LTC": ["litecoin", "ltc"],
            "XRP": ["ripple", "xrp"]
        }
    
    def extract_mentions(self, text: str) -> List[str]:
        """Извлекаем упоминания криптовалют"""
        text_lower = text.lower()
        mentions = []
        
        for symbol, keywords in self.crypto_symbols.items():
            if any(keyword in text_lower for keyword in keywords):
                mentions.append(symbol)
        
        return mentions

class ImpactScorer:
    """Оценка влияния новости на рынок"""
    
    def __init__(self):
        self.impact_keywords = {
            "high": ["breaking", "urgent", "exclusive", "major", "significant"],
            "medium": ["announcement", "update", "partnership", "launch"],
            "low": ["rumor", "speculation", "analysis", "opinion"]
        }
    
    def calculate_impact(self, text: str, source: str, sentiment_score: float) -> float:
        """Рассчитываем оценку влияния новости"""
        text_lower = text.lower()
        
        # Базовый скор по ключевым словам
        base_score = 0.5
        for level, keywords in self.impact_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                if level == "high":
                    base_score = 0.9
                elif level == "medium":
                    base_score = 0.7
                elif level == "low":
                    base_score = 0.3
                break
        
        # Модификатор по источнику
        source_modifier = 1.0
        if source == "tradingview":
            source_modifier = 0.8
        elif source == "investing":
            source_modifier = 0.9
        elif source == "telegram":
            source_modifier = 0.7
        
        # Модификатор по сентименту
        sentiment_modifier = 1.0 + abs(sentiment_score) * 0.3
        
        # Финальный скор
        impact_score = base_score * source_modifier * sentiment_modifier
        return min(1.0, max(0.0, impact_score))

class TelegramNewsCollector:
    """Сборщик новостей из Telegram"""
    
    def __init__(self, api_id: str, api_hash: str, phone: str):
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone = phone
        self.client = None
    
    async def connect(self):
        """Подключаемся к Telegram"""
        try:
            from telethon import TelegramClient
            self.client = TelegramClient('crypto_news_session', self.api_id, self.api_hash)
            await self.client.start(phone=self.phone)
            logger.info("Telegram клиент подключен")
            return True
        except Exception as e:
            logger.error(f"Ошибка подключения к Telegram: {e}")
            return False
    
    async def collect_news(self, channels: List[str], limit: int = 100) -> List[NewsItem]:
        """Собираем новости из каналов"""
        if not self.client:
            return []
        
        news_items = []
        
        for channel in channels:
            try:
                entity = await self.client.get_entity(channel)
                messages = await self.client.get_messages(entity, limit=limit)
                
                for msg in messages:
                    if msg.text and len(msg.text) > 20:
                        news_item = NewsItem(
                            id=f"tg_{msg.id}",
                            source="telegram",
                            channel=channel,
                            title=msg.text[:100] + "..." if len(msg.text) > 100 else msg.text,
                            text=msg.text,
                            url="",
                            date_utc=msg.date,
                            sentiment_score=0.0,
                            sentiment_label="NEUTRAL",
                            category="general",
                            impact_score=0.5,
                            crypto_mentions=[]
                        )
                        news_items.append(news_item)
                
            except Exception as e:
                logger.error(f"Ошибка сбора новостей из {channel}: {e}")
        
        return news_items

class TradingViewCollector:
    """Сборщик новостей из TradingView"""
    
    def __init__(self):
        self.base_url = "https://www.tradingview.com"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    
    def collect_news(self, limit: int = 50) -> List[NewsItem]:
        """Собираем новости из TradingView"""
        try:
            # TradingView API endpoint для новостей
            url = f"{self.base_url}/news/"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"TradingView вернул статус {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            news_items = []
            
            # Парсим новости (структура может меняться)
            news_elements = soup.find_all('div', class_='tv-news-item')
            
            for i, element in enumerate(news_elements[:limit]):
                try:
                    title_elem = element.find('a', class_='tv-news-item__title')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    url = self.base_url + title_elem.get('href', '')
                    
                    # Получаем полный текст новости
                    full_text = self._get_full_text(url)
                    
                    news_item = NewsItem(
                        id=f"tv_{i}",
                        source="tradingview",
                        channel="tradingview",
                        title=title,
                        text=full_text or title,
                        url=url,
                        date_utc=datetime.utcnow() - timedelta(hours=i),
                        sentiment_score=0.0,
                        sentiment_label="NEUTRAL",
                        category="general",
                        impact_score=0.5,
                        crypto_mentions=[]
                    )
                    news_items.append(news_item)
                    
                except Exception as e:
                    logger.error(f"Ошибка парсинга новости TradingView: {e}")
            
            return news_items
            
        except Exception as e:
            logger.error(f"Ошибка сбора новостей TradingView: {e}")
            return []
    
    def _get_full_text(self, url: str) -> str:
        """Получаем полный текст новости"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                content = soup.find('div', class_='tv-news-item__content')
                if content:
                    return content.get_text(strip=True)
        except Exception as e:
            logger.warning(f"Не удалось получить полный текст: {e}")
        return ""

class InvestingCollector:
    """Сборщик новостей из Investing.com"""
    
    def __init__(self):
        self.base_url = "https://www.investing.com"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    
    def collect_news(self, limit: int = 50) -> List[NewsItem]:
        """Собираем новости из Investing.com"""
        try:
            url = f"{self.base_url}/cryptocurrency-news/"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Investing.com вернул статус {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            news_items = []
            
            # Парсим новости
            news_elements = soup.find_all('article', class_='js-article-item')
            
            for i, element in enumerate(news_elements[:limit]):
                try:
                    title_elem = element.find('a', class_='title')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    url = self.base_url + title_elem.get('href', '')
                    
                    # Получаем полный текст
                    full_text = self._get_full_text(url)
                    
                    news_item = NewsItem(
                        id=f"inv_{i}",
                        source="investing",
                        channel="investing",
                        title=title,
                        text=full_text or title,
                        url=url,
                        date_utc=datetime.utcnow() - timedelta(hours=i),
                        sentiment_score=0.0,
                        sentiment_label="NEUTRAL",
                        category="general",
                        impact_score=0.5,
                        crypto_mentions=[]
                    )
                    news_items.append(news_item)
                    
                except Exception as e:
                    logger.error(f"Ошибка парсинга новости Investing: {e}")
            
            return news_items
            
        except Exception as e:
            logger.error(f"Ошибка сбора новостей Investing: {e}")
            return []
    
    def _get_full_text(self, url: str) -> str:
        """Получаем полный текст новости"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                content = soup.find('div', class_='WYSIWYG articlePage')
                if content:
                    return content.get_text(strip=True)
        except Exception as e:
            logger.warning(f"Не удалось получить полный текст: {e}")
        return ""

class EnhancedNewsManager:
    """Главный менеджер новостей"""
    
    def __init__(self, telegram_config: Optional[Dict] = None):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.news_classifier = NewsClassifier()
        self.crypto_extractor = CryptoMentionExtractor()
        self.impact_scorer = ImpactScorer()
        
        self.telegram_collector = None
        if telegram_config:
            self.telegram_collector = TelegramNewsCollector(
                telegram_config.get('api_id'),
                telegram_config.get('api_hash'),
                telegram_config.get('phone')
            )
        
        self.tradingview_collector = TradingViewCollector()
        self.investing_collector = InvestingCollector()
    
    async def collect_all_news(self, telegram_channels: List[str] = None) -> List[NewsItem]:
        """Собираем новости из всех источников"""
        all_news = []
        
        # Telegram новости
        if self.telegram_collector and telegram_channels:
            try:
                await self.telegram_collector.connect()
                tg_news = await self.telegram_collector.collect_news(telegram_channels)
                all_news.extend(tg_news)
                logger.info(f"Собрано {len(tg_news)} новостей из Telegram")
            except Exception as e:
                logger.error(f"Ошибка сбора Telegram новостей: {e}")
        
        # TradingView новости
        try:
            tv_news = self.tradingview_collector.collect_news()
            all_news.extend(tv_news)
            logger.info(f"Собрано {len(tv_news)} новостей из TradingView")
        except Exception as e:
            logger.error(f"Ошибка сбора TradingView новостей: {e}")
        
        # Investing.com новости
        try:
            inv_news = self.investing_collector.collect_news()
            all_news.extend(inv_news)
            logger.info(f"Собрано {len(inv_news)} новостей из Investing.com")
        except Exception as e:
            logger.error(f"Ошибка сбора Investing новостей: {e}")
        
        return all_news
    
    def analyze_news(self, news_items: List[NewsItem]) -> List[NewsItem]:
        """Анализируем все новости"""
        analyzed_news = []
        
        for item in news_items:
            try:
                # Анализ сентимента
                sentiment_result = self.sentiment_analyzer.analyze_sentiment(item.text)
                item.sentiment_score = sentiment_result['combined_score']
                item.sentiment_label = sentiment_result['label']
                
                # Классификация
                item.category = self.news_classifier.classify_news(item.text)
                
                # Извлечение упоминаний криптовалют
                item.crypto_mentions = self.crypto_extractor.extract_mentions(item.text)
                
                # Оценка влияния
                item.impact_score = self.impact_scorer.calculate_impact(
                    item.text, item.source, item.sentiment_score
                )
                
                analyzed_news.append(item)
                
            except Exception as e:
                logger.error(f"Ошибка анализа новости {item.id}: {e}")
        
        return analyzed_news
    
    def get_market_sentiment(self, news_items: List[NewsItem]) -> Dict[str, Any]:
        """Получаем общий сентимент рынка"""
        if not news_items:
            return {"overall_sentiment": 0.0, "sentiment_label": "NEUTRAL"}
        
        # Взвешенный сентимент по влиянию новостей
        weighted_sentiment = sum(
            item.sentiment_score * item.impact_score for item in news_items
        ) / sum(item.impact_score for item in news_items)
        
        # Определяем лейбл
        if weighted_sentiment >= 0.1:
            sentiment_label = "BULLISH"
        elif weighted_sentiment <= -0.1:
            sentiment_label = "BEARISH"
        else:
            sentiment_label = "NEUTRAL"
        
        return {
            "overall_sentiment": weighted_sentiment,
            "sentiment_label": sentiment_label,
            "total_news": len(news_items),
            "positive_news": len([n for n in news_items if n.sentiment_label == "POSITIVE"]),
            "negative_news": len([n for n in news_items if n.sentiment_label == "NEGATIVE"]),
            "neutral_news": len([n for n in news_items if n.sentiment_label == "NEUTRAL"])
        }

# Пример использования
async def main():
    """Пример использования расширенного новостного модуля"""
    
    # Конфигурация Telegram (заполните своими данными)
    telegram_config = {
        'api_id': 'YOUR_API_ID',
        'api_hash': 'YOUR_API_HASH',
        'phone': 'YOUR_PHONE'
    }
    
    # Инициализация менеджера
    news_manager = EnhancedNewsManager(telegram_config)
    
    # Каналы Telegram для мониторинга
    telegram_channels = [
        "binanceupdates",
        "cryptocom",
        "coinbase",
        "kraken"
    ]
    
    # Сбор новостей
    print("🔍 Собираем новости...")
    news_items = await news_manager.collect_all_news(telegram_channels)
    
    # Анализ новостей
    print("📊 Анализируем новости...")
    analyzed_news = news_manager.analyze_news(news_items)
    
    # Общий сентимент рынка
    market_sentiment = news_manager.get_market_sentiment(analyzed_news)
    
    print(f"📈 Сентимент рынка: {market_sentiment['sentiment_label']}")
    print(f"📊 Общий скор: {market_sentiment['overall_sentiment']:.3f}")
    print(f"📰 Всего новостей: {market_sentiment['total_news']}")
    
    # Сохраняем результаты
    news_df = pd.DataFrame([
        {
            'id': item.id,
            'source': item.source,
            'title': item.title,
            'sentiment_score': item.sentiment_score,
            'sentiment_label': item.sentiment_label,
            'category': item.category,
            'impact_score': item.impact_score,
            'crypto_mentions': ', '.join(item.crypto_mentions),
            'date': item.date_utc
        }
        for item in analyzed_news
    ])
    
    news_df.to_csv('AI/data/enhanced_news.csv', index=False)
    print(f"💾 Новости сохранены в AI/data/enhanced_news.csv")

if __name__ == "__main__":
    asyncio.run(main())
