#!/usr/bin/env python3
"""
Улучшенная нейросеть для анализа криптовалют
- Transformer архитектура
- Мультимодальный анализ (цена + новости + индикаторы)
- Attention механизмы
- Ensemble методы
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Конфигурация модели"""
    # Размеры входов
    price_features: int = 8
    news_features: int = 64
    indicator_features: int = 26
    
    # Transformer параметры
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_seq_length: int = 100
    
    # Классификация
    num_classes: int = 3  # DOWN, NEUTRAL, UP
    use_news: bool = True
    use_indicators: bool = True
    
    # Обучение
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000

class PositionalEncoding(nn.Module):
    """Позиционное кодирование для Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Многоголовое внимание с улучшенной архитектурой"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        batch_size = query.shape[0]
        
        # Линейные преобразования
        Q = self.w_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(query.device)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Применяем внимание
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.w_o(out)
        
        return out, attention

class TransformerBlock(nn.Module):
    """Блок Transformer с улучшенной архитектурой"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # Self-attention
        attn_out, attention = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feedforward
        ff_out = self.feedforward(x)
        x = self.norm2(x + ff_out)
        
        return x, attention

class FeatureEncoder(nn.Module):
    """Кодировщик признаков для разных типов данных"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Кодировщик ценовых данных
        self.price_encoder = nn.Sequential(
            nn.Linear(config.price_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, config.d_model // 2)
        )
        
        # Кодировщик новостей (если используется)
        if config.use_news:
            self.news_encoder = nn.Sequential(
                nn.Linear(config.news_features, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, config.d_model // 2)
            )
        
        # Кодировщик технических индикаторов
        if config.use_indicators:
            self.indicator_encoder = nn.Sequential(
                nn.Linear(config.indicator_features, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, config.d_model // 2)
            )
        
        # Проекция в общее пространство признаков
        self.feature_projection = nn.Linear(config.d_model, config.d_model)
    
    def forward(self, price_data: Tensor, news_data: Optional[Tensor] = None, 
                indicators: Optional[Tensor] = None) -> Tensor:
        batch_size = price_data.shape[0]
        
        # Кодируем ценовые данные
        price_encoded = self.price_encoder(price_data)
        
        # Кодируем новости
        if self.config.use_news and news_data is not None:
            news_encoded = self.news_encoder(news_data)
        else:
            news_encoded = torch.zeros(batch_size, self.config.d_model // 2, device=price_data.device)
        
        # Кодируем индикаторы
        if self.config.use_indicators and indicators is not None:
            indicator_encoded = self.indicator_encoder(indicators)
        else:
            indicator_encoded = torch.zeros(batch_size, self.config.d_model // 2, device=price_data.device)
        
        # Объединяем все признаки
        combined_features = torch.cat([price_encoded, news_encoded, indicator_encoded], dim=-1)
        
        # Проецируем в общее пространство
        encoded_features = self.feature_projection(combined_features)
        
        return encoded_features

class CryptoTransformer(nn.Module):
    """Основная модель для анализа криптовалют"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Кодировщик признаков
        self.feature_encoder = FeatureEncoder(config)
        
        # Позиционное кодирование
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_length)
        
        # Transformer блоки
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.nhead, config.dim_feedforward, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # Выходные слои
        self.output_projection = nn.Linear(config.d_model, config.d_model)
        self.classifier = nn.Linear(config.d_model, config.num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов модели"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def create_padding_mask(self, seq_lengths: List[int], max_len: int) -> Tensor:
        """Создаем маску для padding"""
        batch_size = len(seq_lengths)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        
        for i, length in enumerate(seq_lengths):
            mask[i, length:] = True
        
        return mask
    
    def forward(self, price_data: Tensor, news_data: Optional[Tensor] = None,
                indicators: Optional[Tensor] = None, seq_lengths: Optional[List[int]] = None) -> Dict[str, Tensor]:
        batch_size, seq_len, _ = price_data.shape
        
        # Кодируем признаки
        encoded_features = self.feature_encoder(price_data, news_data, indicators)
        
        # Добавляем позиционное кодирование
        encoded_features = encoded_features.transpose(0, 1)  # (seq_len, batch, d_model)
        encoded_features = self.pos_encoding(encoded_features)
        encoded_features = encoded_features.transpose(0, 1)  # (batch, seq_len, d_model)
        
        # Создаем маску для padding
        if seq_lengths is not None:
            padding_mask = self.create_padding_mask(seq_lengths, seq_len)
        else:
            padding_mask = None
        
        # Проходим через Transformer блоки
        attention_weights = []
        x = encoded_features
        
        for transformer_block in self.transformer_blocks:
            x, attention = transformer_block(x, padding_mask)
            attention_weights.append(attention)
        
        # Глобальное среднее по последовательности
        if padding_mask is not None:
            # Учитываем только валидные позиции
            mask_expanded = padding_mask.unsqueeze(-1).expand_as(x)
            x_masked = x.masked_fill(mask_expanded, 0.0)
            seq_lengths_tensor = torch.tensor(seq_lengths, dtype=torch.float, device=x.device)
            x_pooled = x_masked.sum(dim=1) / seq_lengths_tensor.unsqueeze(-1)
        else:
            x_pooled = x.mean(dim=1)
        
        # Выходные слои
        x_projected = self.output_projection(x_pooled)
        x_projected = self.dropout(x_projected)
        
        logits = self.classifier(x_projected)
        probs = F.softmax(logits, dim=-1)
        
        return {
            "logits": logits,
            "probs": probs,
            "attention_weights": attention_weights,
            "encoded_features": encoded_features
        }

class EnsembleModel(nn.Module):
    """Ensemble модель, объединяющая несколько подходов"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Основная Transformer модель
        self.transformer_model = CryptoTransformer(config)
        
        # Дополнительные модели для ensemble
        self.lstm_model = nn.LSTM(
            input_size=config.d_model,
            hidden_size=128,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )
        
        self.gru_model = nn.GRU(
            input_size=config.d_model,
            hidden_size=128,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )
        
        # Ensemble классификатор
        self.ensemble_classifier = nn.Sequential(
            nn.Linear(config.num_classes * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, config.num_classes)
        )
    
    def forward(self, price_data: Tensor, news_data: Optional[Tensor] = None,
                indicators: Optional[Tensor] = None, seq_lengths: Optional[List[int]] = None) -> Dict[str, Tensor]:
        
        # Transformer предсказания
        transformer_output = self.transformer_model(price_data, news_data, indicators, seq_lengths)
        transformer_probs = transformer_output["probs"]
        
        # LSTM предсказания
        encoded_features = transformer_output["encoded_features"]
        lstm_out, _ = self.lstm_model(encoded_features)
        lstm_pooled = lstm_out.mean(dim=1)
        lstm_logits = nn.Linear(lstm_pooled.shape[-1], self.config.num_classes)(lstm_pooled)
        lstm_probs = F.softmax(lstm_logits, dim=-1)
        
        # GRU предсказания
        gru_out, _ = self.gru_model(encoded_features)
        gru_pooled = gru_out.mean(dim=1)
        gru_logits = nn.Linear(gru_pooled.shape[-1], self.config.num_classes)(gru_pooled)
        gru_probs = F.softmax(gru_logits, dim=-1)
        
        # Объединяем предсказания
        combined_probs = torch.cat([transformer_probs, lstm_probs, gru_probs], dim=-1)
        ensemble_logits = self.ensemble_classifier(combined_probs)
        ensemble_probs = F.softmax(ensemble_logits, dim=-1)
        
        return {
            "transformer_probs": transformer_probs,
            "lstm_probs": lstm_probs,
            "gru_probs": gru_probs,
            "ensemble_probs": ensemble_probs,
            "ensemble_logits": ensemble_logits,
            "attention_weights": transformer_output["attention_weights"]
        }

class CryptoAnalyzer:
    """Высокоуровневый API для анализа криптовалют"""
    
    def __init__(self, config: ModelConfig, model_path: Optional[str] = None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Инициализируем модель
        self.model = EnsembleModel(config).to(self.device)
        
        # Загружаем предобученную модель если указан путь
        if model_path and torch.cuda.is_available():
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Модель загружена из {model_path}")
            except Exception as e:
                print(f"Ошибка загрузки модели: {e}")
        
        self.model.eval()
    
    def preprocess_data(self, price_data: np.ndarray, news_data: Optional[np.ndarray] = None,
                       indicators: Optional[np.ndarray] = None) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Предобработка данных для модели"""
        
        # Преобразуем в тензоры
        price_tensor = torch.FloatTensor(price_data).to(self.device)
        
        news_tensor = None
        if news_data is not None and self.config.use_news:
            news_tensor = torch.FloatTensor(news_data).to(self.device)
        
        indicators_tensor = None
        if indicators is not None and self.config.use_indicators:
            indicators_tensor = torch.FloatTensor(indicators).to(self.device)
        
        return price_tensor, news_tensor, indicators_tensor
    
    def predict(self, price_data: np.ndarray, news_data: Optional[np.ndarray] = None,
                indicators: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Делаем предсказание"""
        
        with torch.no_grad():
            # Предобработка
            price_tensor, news_tensor, indicators_tensor = self.preprocess_data(
                price_data, news_data, indicators
            )
            
            # Предсказание
            output = self.model(price_tensor, news_tensor, indicators_tensor)
            
            # Извлекаем результаты
            ensemble_probs = output["ensemble_probs"].cpu().numpy()[0]
            transformer_probs = output["transformer_probs"].cpu().numpy()[0]
            
            # Определяем класс
            predicted_class = np.argmax(ensemble_probs)
            confidence = ensemble_probs[predicted_class]
            
            # Маппинг классов
            class_mapping = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
            prediction = class_mapping[predicted_class]
            
            return {
                "prediction": prediction,
                "confidence": float(confidence),
                "probabilities": {
                    "down": float(ensemble_probs[0]),
                    "neutral": float(ensemble_probs[1]),
                    "up": float(ensemble_probs[2])
                },
                "transformer_probs": {
                    "down": float(transformer_probs[0]),
                    "neutral": float(transformer_probs[1]),
                    "up": float(transformer_probs[2])
                },
                "attention_weights": [w.cpu().numpy() for w in output["attention_weights"]]
            }
    
    def analyze_market(self, price_data: np.ndarray, news_data: Optional[np.ndarray] = None,
                      indicators: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Комплексный анализ рынка"""
        
        # Базовое предсказание
        prediction_result = self.predict(price_data, news_data, indicators)
        
        # Дополнительный анализ
        analysis = {
            "prediction": prediction_result,
            "market_analysis": {
                "trend_strength": self._calculate_trend_strength(price_data),
                "volatility": self._calculate_volatility(price_data),
                "support_resistance": self._find_support_resistance(price_data),
                "momentum": self._calculate_momentum(price_data)
            },
            "risk_assessment": {
                "risk_level": self._assess_risk(prediction_result, price_data),
                "confidence_interval": self._calculate_confidence_interval(prediction_result),
                "recommendation": self._generate_recommendation(prediction_result)
            }
        }
        
        return analysis
    
    def _calculate_trend_strength(self, price_data: np.ndarray) -> float:
        """Рассчитываем силу тренда"""
        if len(price_data) < 20:
            return 0.0
        
        # Используем последние 20 точек для расчета тренда
        recent_prices = price_data[-20:, 3]  # Close prices
        
        # Линейная регрессия для определения тренда
        x = np.arange(len(recent_prices))
        slope, _ = np.polyfit(x, recent_prices, 1)
        
        # Нормализуем наклон
        price_range = recent_prices.max() - recent_prices.min()
        if price_range == 0:
            return 0.0
        
        trend_strength = slope / price_range * 100
        return np.clip(trend_strength, -1.0, 1.0)
    
    def _calculate_volatility(self, price_data: np.ndarray) -> float:
        """Рассчитываем волатильность"""
        if len(price_data) < 20:
            return 0.0
        
        returns = np.diff(price_data[:, 3]) / price_data[:-1, 3]  # Returns
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        return float(volatility)
    
    def _find_support_resistance(self, price_data: np.ndarray) -> Dict[str, float]:
        """Находим уровни поддержки и сопротивления"""
        if len(price_data) < 50:
            return {"support": 0.0, "resistance": 0.0}
        
        recent_prices = price_data[-50:, 3]  # Close prices
        
        # Простой алгоритм поиска уровней
        support = np.percentile(recent_prices, 25)
        resistance = np.percentile(recent_prices, 75)
        
        return {
            "support": float(support),
            "resistance": float(resistance)
        }
    
    def _calculate_momentum(self, price_data: np.ndarray) -> float:
        """Рассчитываем момент"""
        if len(price_data) < 14:
            return 0.0
        
        # RSI-like momentum
        recent_prices = price_data[-14:, 3]  # Close prices
        gains = np.maximum(np.diff(recent_prices), 0)
        losses = np.maximum(-np.diff(recent_prices), 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 1.0
        
        rs = avg_gain / avg_loss
        momentum = 100 - (100 / (1 + rs))
        
        return float(momentum)
    
    def _assess_risk(self, prediction_result: Dict, price_data: np.ndarray) -> str:
        """Оцениваем уровень риска"""
        confidence = prediction_result["confidence"]
        volatility = self._calculate_volatility(price_data)
        
        if confidence < 0.6 or volatility > 0.8:
            return "HIGH"
        elif confidence < 0.8 or volatility > 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_confidence_interval(self, prediction_result: Dict) -> Dict[str, float]:
        """Рассчитываем доверительный интервал"""
        confidence = prediction_result["confidence"]
        
        # Простая формула для доверительного интервала
        margin = (1 - confidence) * 0.5
        
        return {
            "lower": max(0.0, confidence - margin),
            "upper": min(1.0, confidence + margin)
        }
    
    def _generate_recommendation(self, prediction_result: Dict) -> str:
        """Генерируем рекомендацию"""
        prediction = prediction_result["prediction"]
        confidence = prediction_result["confidence"]
        
        if confidence < 0.6:
            return "WAIT - Недостаточно уверенности для торговли"
        
        if prediction == "UP":
            return "BUY - Рекомендуется покупка"
        elif prediction == "DOWN":
            return "SELL - Рекомендуется продажа"
        else:
            return "HOLD - Рекомендуется удержание позиции"

# Функция для создания модели
def create_enhanced_model(config: ModelConfig) -> CryptoAnalyzer:
    """Создаем улучшенную модель анализа криптовалют"""
    return CryptoAnalyzer(config)

# Пример использования
if __name__ == "__main__":
    # Конфигурация модели
    config = ModelConfig(
        price_features=8,
        news_features=64,
        indicator_features=26,
        d_model=256,
        nhead=8,
        num_layers=6,
        use_news=True,
        use_indicators=True
    )
    
    # Создаем модель
    analyzer = create_enhanced_model(config)
    
    print("🚀 Улучшенная модель анализа криптовалют создана!")
    print(f"📊 Конфигурация: {config}")
    print(f"💻 Устройство: {analyzer.device}")
    
    # Тестовые данные
    test_price_data = np.random.randn(100, 8)  # 100 временных точек, 8 признаков
    test_news_data = np.random.randn(100, 64)  # 100 новостей, 64 признака
    test_indicators = np.random.randn(100, 26)  # 100 индикаторов, 26 признаков
    
    # Тестируем предсказание
    print("\n🔮 Тестируем предсказание...")
    result = analyzer.predict(test_price_data, test_news_data, test_indicators)
    
    print(f"📈 Предсказание: {result['prediction']}")
    print(f"🎯 Уверенность: {result['confidence']:.3f}")
    print(f"📊 Вероятности: {result['probabilities']}")
    
    # Тестируем комплексный анализ
    print("\n📊 Тестируем комплексный анализ...")
    analysis = analyzer.analyze_market(test_price_data, test_news_data, test_indicators)
    
    print(f"📈 Сила тренда: {analysis['market_analysis']['trend_strength']:.3f}")
    print(f"📊 Волатильность: {analysis['market_analysis']['volatility']:.3f}")
    print(f"⚠️ Уровень риска: {analysis['risk_assessment']['risk_level']}")
    print(f"💡 Рекомендация: {analysis['risk_assessment']['recommendation']}")
