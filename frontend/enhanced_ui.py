#!/usr/bin/env python3
"""
Современный Streamlit UI для анализа криптовалют
- Анализ новостей и сентимента
- Технический анализ
- Машинное обучение
- Интерактивные графики
- Дашборд аналитики
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import asyncio
import time
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Добавляем путь к модулям
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Импорты из нашего проекта
try:
    from AI.news.enhanced_news import EnhancedNewsManager
    from AI.models.enhanced_model import create_enhanced_model, ModelConfig
    from AI.features.indicators import compute_indicators, compute_indicators_extended
    from AI.data.generate_synthetic import generate_ohlcv
except ImportError as e:
    st.error(f"Ошибка импорта модулей: {e}")
    st.info("Убедитесь, что все зависимости установлены")

# Настройка страницы
st.set_page_config(
    page_title="Crypto AI Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .news-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .sentiment-positive { color: #28a745; }
    .sentiment-negative { color: #dc3545; }
    .sentiment-neutral { color: #6c757d; }
</style>
""", unsafe_allow_html=True)

class CryptoAnalyticsUI:
    """Главный класс UI для анализа криптовалют"""
    
    def __init__(self):
        self.api_url = os.environ.get("API_URL", "http://localhost:8000")
        self.news_manager = None
        self.model = None
        self.initialize_components()
    
    def initialize_components(self):
        """Инициализируем компоненты"""
        try:
            # Инициализируем новостной менеджер
            self.news_manager = EnhancedNewsManager()
            
            # Инициализируем модель
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
            self.model = create_enhanced_model(config)
            
        except Exception as e:
            st.warning(f"Некоторые компоненты не инициализированы: {e}")
    
    def render_header(self):
        """Отображаем заголовок"""
        st.markdown('<h1 class="main-header">🚀 Crypto AI Analytics</h1>', unsafe_allow_html=True)
        st.markdown("---")
    
    def render_sidebar(self):
        """Отображаем боковую панель"""
        with st.sidebar:
            st.header("⚙️ Настройки")
            
            # Выбор криптовалюты
            crypto_symbol = st.selectbox(
                "Криптовалюта",
                ["BTC", "ETH", "BNB", "ADA", "SOL", "DOT", "DOGE", "MATIC", "LTC", "XRP"],
                index=0
            )
            
            # Временной интервал
            timeframe = st.selectbox(
                "Временной интервал",
                ["1m", "5m", "15m", "1h", "4h", "1d"],
                index=3
            )
            
            # Количество свечей
            num_candles = st.slider("Количество свечей", 100, 1000, 500)
            
            # Источники новостей
            st.subheader("📰 Источники новостей")
            use_telegram = st.checkbox("Telegram", value=True)
            use_tradingview = st.checkbox("TradingView", value=True)
            use_investing = st.checkbox("Investing.com", value=True)
            
            # Настройки модели
            st.subheader("🤖 Настройки модели")
            use_news_analysis = st.checkbox("Анализ новостей", value=True)
            use_technical_analysis = st.checkbox("Технический анализ", value=True)
            ensemble_method = st.checkbox("Ensemble методы", value=True)
            
            # Кнопки управления
            st.subheader("🔄 Управление")
            if st.button("🔄 Обновить данные", type="primary"):
                st.session_state.refresh_data = True
            
            if st.button("📊 Сгенерировать отчет"):
                st.session_state.generate_report = True
            
            return {
                "crypto_symbol": crypto_symbol,
                "timeframe": timeframe,
                "num_candles": num_candles,
                "use_telegram": use_telegram,
                "use_tradingview": use_tradingview,
                "use_investing": use_investing,
                "use_news_analysis": use_news_analysis,
                "use_technical_analysis": use_technical_analysis,
                "ensemble_method": ensemble_method
            }
    
    def render_price_chart(self, data: pd.DataFrame):
        """Отображаем график цены"""
        st.subheader("📈 График цены")
        
        # Создаем candlestick график
        fig = go.Figure(data=[go.Candlestick(
            x=data['datetime'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name="OHLC"
        )])
        
        fig.update_layout(
            title=f"Цена {st.session_state.get('crypto_symbol', 'BTC')}",
            xaxis_title="Время",
            yaxis_title="Цена (USD)",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_technical_indicators(self, data: pd.DataFrame):
        """Отображаем технические индикаторы"""
        st.subheader("📊 Технические индикаторы")
        
        try:
            # Вычисляем индикаторы
            df_with_indicators = compute_indicators_extended(data)
            
            # Создаем подграфики
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=("RSI", "MACD", "Bollinger Bands", "Stochastic", "Volume", "MFI"),
                vertical_spacing=0.1
            )
            
            # RSI
            fig.add_trace(
                go.Scatter(x=df_with_indicators.index, y=df_with_indicators['rsi_14'], name="RSI"),
                row=1, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
            
            # MACD
            fig.add_trace(
                go.Scatter(x=df_with_indicators.index, y=df_with_indicators['macd'], name="MACD"),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=df_with_indicators.index, y=df_with_indicators['macd_signal'], name="Signal"),
                row=1, col=2
            )
            
            # Bollinger Bands
            fig.add_trace(
                go.Scatter(x=df_with_indicators.index, y=df_with_indicators['bb_high'], name="BB High", line=dict(color='gray')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=df_with_indicators.index, y=df_with_indicators['bb_low'], name="BB Low", line=dict(color='gray')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=df_with_indicators.index, y=df_with_indicators['close'], name="Price"),
                row=2, col=1
            )
            
            # Stochastic
            fig.add_trace(
                go.Scatter(x=df_with_indicators.index, y=df_with_indicators['stoch_k'], name="%K"),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=df_with_indicators.index, y=df_with_indicators['stoch_d'], name="%D"),
                row=2, col=2
            )
            
            # Volume
            fig.add_trace(
                go.Bar(x=df_with_indicators.index, y=df_with_indicators['volume'], name="Volume"),
                row=3, col=1
            )
            
            # MFI
            fig.add_trace(
                go.Scatter(x=df_with_indicators.index, y=df_with_indicators['mfi_14'], name="MFI"),
                row=3, col=2
            )
            
            fig.update_layout(height=800, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Ошибка при вычислении индикаторов: {e}")
    
    def render_news_analysis(self, news_data: pd.DataFrame):
        """Отображаем анализ новостей"""
        st.subheader("📰 Анализ новостей")
        
        if news_data.empty:
            st.info("Новости не найдены. Попробуйте обновить данные.")
            return
        
        # Метрики новостей
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Всего новостей", len(news_data))
        
        with col2:
            positive_news = len(news_data[news_data['sentiment_label'] == 'POSITIVE'])
            st.metric("Позитивные", positive_news)
        
        with col3:
            negative_news = len(news_data[news_data['sentiment_label'] == 'NEGATIVE'])
            st.metric("Негативные", negative_news)
        
        with col4:
            avg_impact = news_data['impact_score'].mean()
            st.metric("Средний Impact", f"{avg_impact:.3f}")
        
        # График сентимента
        fig = px.histogram(
            news_data, 
            x='sentiment_score', 
            color='sentiment_label',
            title="Распределение сентимента новостей",
            labels={'sentiment_score': 'Сентимент', 'sentiment_label': 'Лейбл'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Список новостей
        st.subheader("📋 Последние новости")
        
        for _, news in news_data.head(10).iterrows():
            sentiment_class = f"sentiment-{news['sentiment_label'].lower()}"
            
            with st.container():
                st.markdown(f"""
                <div class="news-card">
                    <h4>{news['title']}</h4>
                    <p><strong>Источник:</strong> {news['source']} | <strong>Категория:</strong> {news['category']}</p>
                    <p><strong>Сентимент:</strong> <span class="{sentiment_class}">{news['sentiment_label']} ({news['sentiment_score']:.3f})</span></p>
                    <p><strong>Impact:</strong> {news['impact_score']:.3f}</p>
                    <p><strong>Криптовалюты:</strong> {news['crypto_mentions']}</p>
                    <small>{news['date']}</small>
                </div>
                """, unsafe_allow_html=True)
    
    def render_machine_learning(self, price_data: pd.DataFrame, news_data: pd.DataFrame):
        """Отображаем результаты машинного обучения"""
        st.subheader("🤖 Машинное обучение")
        
        try:
            # Подготавливаем данные для модели
            if len(price_data) < 100:
                st.warning("Недостаточно данных для анализа. Нужно минимум 100 свечей.")
                return
            
            # Создаем тестовые данные
            price_features = price_data[['open', 'high', 'low', 'close', 'volume']].values
            price_features = np.concatenate([
                price_features,
                np.random.randn(len(price_features), 3)  # Дополнительные признаки
            ], axis=1)
            
            # Тестируем модель
            with st.spinner("Анализируем данные..."):
                result = self.model.predict(price_features)
                
                # Отображаем результаты
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Предсказание", result['prediction'])
                
                with col2:
                    st.metric("Уверенность", f"{result['confidence']:.3f}")
                
                with col3:
                    # Определяем цвет для предсказания
                    if result['prediction'] == "UP":
                        color = "green"
                    elif result['prediction'] == "DOWN":
                        color = "red"
                    else:
                        color = "gray"
                    
                    st.markdown(f"""
                    <div style="background-color: {color}; color: white; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                        <h3>{result['prediction']}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                # График вероятностей
                fig = go.Figure(data=[
                    go.Bar(
                        x=['DOWN', 'NEUTRAL', 'UP'],
                        y=[result['probabilities']['down'], result['probabilities']['neutral'], result['probabilities']['up']],
                        marker_color=['red', 'gray', 'green']
                    )
                ])
                
                fig.update_layout(
                    title="Вероятности предсказания",
                    xaxis_title="Класс",
                    yaxis_title="Вероятность",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Детальная информация
                st.subheader("📊 Детали предсказания")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Ensemble вероятности:**")
                    for key, value in result['probabilities'].items():
                        st.write(f"- {key.upper()}: {value:.3f}")
                
                with col2:
                    st.write("**Transformer вероятности:**")
                    for key, value in result['transformer_probs'].items():
                        st.write(f"- {key.upper()}: {value:.3f}")
        
        except Exception as e:
            st.error(f"Ошибка при анализе машинного обучения: {e}")
    
    def render_market_analysis(self, price_data: pd.DataFrame):
        """Отображаем анализ рынка"""
        st.subheader("📊 Анализ рынка")
        
        try:
            # Рассчитываем метрики
            returns = price_data['close'].pct_change().dropna()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                volatility = returns.std() * np.sqrt(252)
                st.metric("Волатильность (годовая)", f"{volatility:.2%}")
            
            with col2:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
            
            with col3:
                max_drawdown = (price_data['close'] / price_data['close'].cummax() - 1).min()
                st.metric("Max Drawdown", f"{max_drawdown:.2%}")
            
            with col4:
                current_price = price_data['close'].iloc[-1]
                price_change = (current_price / price_data['close'].iloc[0] - 1)
                st.metric("Изменение цены", f"{price_change:.2%}")
            
            # График доходности
            cumulative_returns = (1 + returns).cumprod()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values,
                mode='lines',
                name='Кумулятивная доходность'
            ))
            
            fig.update_layout(
                title="Кумулятивная доходность",
                xaxis_title="Время",
                yaxis_title="Доходность",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Ошибка при анализе рынка: {e}")
    
    def render_portfolio_optimization(self):
        """Отображаем оптимизацию портфеля"""
        st.subheader("💼 Оптимизация портфеля")
        
        # Простой калькулятор портфеля
        st.write("**Калькулятор распределения портфеля**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            btc_allocation = st.slider("BTC Allocation (%)", 0, 100, 40)
            eth_allocation = st.slider("ETH Allocation (%)", 0, 100, 30)
            other_allocation = st.slider("Other Allocation (%)", 0, 100, 30)
        
        with col2:
            total_allocation = btc_allocation + eth_allocation + other_allocation
            
            if total_allocation != 100:
                st.warning(f"Общая аллокация: {total_allocation}% (должна быть 100%)")
            else:
                st.success("✅ Аллокация корректна")
            
            # Рассчитываем ожидаемую доходность
            btc_return = 0.15  # Примерные ожидаемые доходности
            eth_return = 0.20
            other_return = 0.10
            
            expected_return = (btc_allocation * btc_return + eth_allocation * eth_return + other_allocation * other_return) / 100
            
            st.metric("Ожидаемая доходность", f"{expected_return:.2%}")
        
        # График распределения
        fig = go.Figure(data=[go.Pie(
            labels=['BTC', 'ETH', 'Other'],
            values=[btc_allocation, eth_allocation, other_allocation],
            hole=0.3
        )])
        
        fig.update_layout(title="Распределение портфеля")
        st.plotly_chart(fig, use_container_width=True)
    
    def render_risk_management(self):
        """Отображаем управление рисками"""
        st.subheader("⚠️ Управление рисками")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Калькулятор позиции**")
            
            account_size = st.number_input("Размер счета (USD)", min_value=1000, value=10000, step=1000)
            risk_per_trade = st.slider("Риск на сделку (%)", 1, 5, 2)
            stop_loss = st.slider("Stop Loss (%)", 5, 20, 10)
            
            risk_amount = account_size * risk_per_trade / 100
            position_size = risk_amount / (stop_loss / 100)
            
            st.metric("Размер позиции", f"${position_size:.2f}")
            st.metric("Риск на сделку", f"${risk_amount:.2f}")
        
        with col2:
            st.write("**Риск-метрики**")
            
            # Примерные метрики
            var_95 = 0.02  # Value at Risk 95%
            max_loss = 0.15  # Максимальный убыток
            recovery_time = 30  # Время восстановления в днях
            
            st.metric("VaR (95%)", f"{var_95:.2%}")
            st.metric("Макс. убыток", f"{max_loss:.2%}")
            st.metric("Время восстановления", f"{recovery_time} дней")
    
    def render_news_sentiment_timeline(self, news_data: pd.DataFrame):
        """Отображаем временную линию сентимента новостей"""
        st.subheader("📅 Временная линия сентимента")
        
        if news_data.empty:
            st.info("Нет данных для отображения")
            return
        
        try:
            # Группируем по времени и рассчитываем средний сентимент
            news_data['date'] = pd.to_datetime(news_data['date'])
            news_data['hour'] = news_data['date'].dt.hour
            
            hourly_sentiment = news_data.groupby('hour')['sentiment_score'].mean().reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hourly_sentiment['hour'],
                y=hourly_sentiment['sentiment_score'],
                mode='lines+markers',
                name='Средний сентимент',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title="Сентимент новостей по часам",
                xaxis_title="Час дня",
                yaxis_title="Сентимент",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Ошибка при создании временной линии: {e}")
    
    def render_correlation_analysis(self, price_data: pd.DataFrame, news_data: pd.DataFrame):
        """Отображаем анализ корреляции"""
        st.subheader("🔗 Анализ корреляции")
        
        try:
            # Рассчитываем корреляцию между ценой и сентиментом
            if not news_data.empty and len(price_data) >= len(news_data):
                # Приводим к одинаковой длине
                min_len = min(len(price_data), len(news_data))
                
                price_returns = price_data['close'].pct_change().dropna().iloc[-min_len:]
                news_sentiment = news_data['sentiment_score'].iloc[-min_len:]
                
                # Рассчитываем корреляцию
                correlation = np.corrcoef(price_returns, news_sentiment)[0, 1]
                
                st.metric("Корреляция цена-сентимент", f"{correlation:.3f}")
                
                # График корреляции
                fig = make_subplots(rows=2, cols=1, subplot_titles=("Изменение цены", "Сентимент новостей"))
                
                fig.add_trace(
                    go.Scatter(x=price_returns.index, y=price_returns.values, name="Изменение цены"),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=news_sentiment.index, y=news_sentiment.values, name="Сентимент"),
                    row=2, col=1
                )
                
                fig.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Ошибка при анализе корреляции: {e}")
    
    def render_export_section(self, price_data: pd.DataFrame, news_data: pd.DataFrame):
        """Отображаем секцию экспорта"""
        st.subheader("💾 Экспорт данных")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Экспорт ценовых данных"):
                csv = price_data.to_csv(index=False)
                st.download_button(
                    label="Скачать CSV",
                    data=csv,
                    file_name=f"crypto_prices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if not news_data.empty and st.button("📰 Экспорт новостей"):
                csv = news_data.to_csv(index=False)
                st.download_button(
                    label="Скачать CSV",
                    data=csv,
                    file_name=f"crypto_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    def run(self):
        """Запускаем UI"""
        # Инициализация сессии
        if 'refresh_data' not in st.session_state:
            st.session_state.refresh_data = False
        if 'generate_report' not in st.session_state:
            st.session_state.generate_report = False
        
        # Отображаем заголовок
        self.render_header()
        
        # Отображаем боковую панель
        settings = self.render_sidebar()
        
        # Основной контент
        if st.session_state.refresh_data or st.button("🔄 Обновить данные"):
            st.session_state.refresh_data = False
            
            with st.spinner("Обновляем данные..."):
                # Генерируем синтетические данные
                price_data = generate_ohlcv(n=settings['num_candles'])
                
                # Создаем тестовые новости
                news_data = pd.DataFrame({
                    'id': range(20),
                    'source': np.random.choice(['telegram', 'tradingview', 'investing'], 20),
                    'title': [f"Тестовая новость {i}" for i in range(20)],
                    'sentiment_score': np.random.normal(0, 0.5, 20),
                    'sentiment_label': np.random.choice(['POSITIVE', 'NEGATIVE', 'NEUTRAL'], 20),
                    'category': np.random.choice(['regulation', 'adoption', 'technology', 'market'], 20),
                    'impact_score': np.random.uniform(0.1, 1.0, 20),
                    'crypto_mentions': ['BTC, ETH'] * 20,
                    'date': [datetime.now() - timedelta(hours=i) for i in range(20)]
                })
                
                # Сохраняем в сессии
                st.session_state.price_data = price_data
                st.session_state.news_data = news_data
                
                st.success("✅ Данные обновлены!")
        
        # Отображаем данные если они есть
        if 'price_data' in st.session_state and 'news_data' in st.session_state:
            price_data = st.session_state.price_data
            news_data = st.session_state.news_data
            
            # Создаем вкладки
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📈 Цена", "📊 Технический анализ", "📰 Новости", "🤖 ML", 
                "📊 Анализ рынка", "💼 Портфель", "⚠️ Риски"
            ])
            
            with tab1:
                self.render_price_chart(price_data)
            
            with tab2:
                self.render_technical_indicators(price_data)
            
            with tab3:
                self.render_news_analysis(news_data)
                self.render_news_sentiment_timeline(news_data)
                self.render_correlation_analysis(price_data, news_data)
            
            with tab4:
                self.render_machine_learning(price_data, news_data)
            
            with tab5:
                self.render_market_analysis(price_data)
            
            with tab6:
                self.render_portfolio_optimization()
            
            with tab7:
                self.render_risk_management()
            
            # Экспорт
            self.render_export_section(price_data, news_data)
        
        else:
            st.info("👆 Нажмите 'Обновить данные' для начала работы")

def main():
    """Главная функция"""
    try:
        ui = CryptoAnalyticsUI()
        ui.run()
    except Exception as e:
        st.error(f"Критическая ошибка: {e}")
        st.info("Проверьте, что все модули установлены и API запущен")

if __name__ == "__main__":
    main()
