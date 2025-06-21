"""
strategy.py
200% Advanced Trading Strategy for the Advanced Stock Trading System.
Includes abstract base class, multi-factor, regime-switching, RL hooks, pluggable interface, backtest, explainability, and reporting.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any
from abc import ABC, abstractmethod
import logging
from .data import DataCollector

# Set up logging for the module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Abstract base class for strategies
class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> Optional[Dict]:
        pass
    @abstractmethod
    def explain(self, data: pd.DataFrame, symbol: str) -> Dict:
        pass
    @abstractmethod
    def report(self, symbol: str = None) -> None:
        pass

# Advanced trading strategy
class TradingStrategy(BaseStrategy):
    """
    200% Advanced Trading Strategy: multi-factor, regime-switching, RL hooks, explainability, reporting.
    """
    def __init__(self, model, scaler, feature_scaler, config: dict = None):
        self.model = model
        self.scaler = scaler
        self.feature_scaler = feature_scaler
        self.signals = []
        self.data_collector = DataCollector()
        self.config = config or {}
        self.regime = 'normal'  # Could be 'bull', 'bear', 'sideways', etc.
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> Optional[Dict]:
        if len(data) < 60:
            return None
        # Multi-factor: combine ML, technical, sentiment, macro
        ml_signal = self._ml_signal(data, symbol)
        tech_signal = self._technical_signal(data)
        sentiment_signal = self._sentiment_signal(data)
        macro_signal = self._macro_signal()
        # Regime switching logic (stub)
        self.regime = self._detect_regime(data)
        # Combine signals (weighted)
        weights = self.config.get('weights', {'ml': 0.5, 'tech': 0.2, 'sentiment': 0.2, 'macro': 0.1})
        combined_score = (
            weights['ml'] * ml_signal['score'] +
            weights['tech'] * tech_signal['score'] +
            weights['sentiment'] * sentiment_signal['score'] +
            weights['macro'] * macro_signal['score']
        )
        final_signal = 'BUY' if combined_score > 0.05 else 'SELL' if combined_score < -0.05 else 'HOLD'
        signal_data = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'signal': final_signal,
            'score': combined_score,
            'regime': self.regime,
            'details': {
                'ml': ml_signal,
                'tech': tech_signal,
                'sentiment': sentiment_signal,
                'macro': macro_signal
            }
        }
        self.signals.append(signal_data)
        return signal_data
    def _ml_signal(self, data: pd.DataFrame, symbol: str) -> Dict:
        # Use last 60 points for ML prediction
        recent_data = data.tail(60)
        processed_data = self.data_collector.calculate_technical_indicators(recent_data)
        feature_columns = [
            'Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'RSI', 'MACD', 'MACD_signal', 'STOCH_K', 'STOCH_D',
            'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'OBV', 'AD',
            'ADX', 'CCI', 'Price_Change', 'Volume_Change',
            'High_Low_Ratio', 'Open_Close_Ratio'
        ]
        available_features = [col for col in feature_columns if col in processed_data.columns]
        feature_data = processed_data[available_features].fillna(method='ffill').fillna(0)
        scaled_features = self.feature_scaler.transform(feature_data.values[-60:])
        X_pred = scaled_features.reshape(1, 60, -1)
        prediction = self.model.predict(X_pred)
        predicted_price = self.scaler.inverse_transform(prediction)[0][0]
        current_price = data['Close'].iloc[-1]
        price_change_pct = (predicted_price - current_price) / current_price
        score = price_change_pct
        return {'score': score, 'predicted_price': predicted_price, 'current_price': current_price}
    def _technical_signal(self, data: pd.DataFrame) -> Dict:
        # Simple technical: RSI, MACD, MA
        latest = data.iloc[-1]
        score = 0.0
        if 'RSI' in latest and latest['RSI'] < 30:
            score += 0.1
        if 'MACD' in latest and 'MACD_signal' in latest and latest['MACD'] > latest['MACD_signal']:
            score += 0.1
        if 'SMA_20' in latest and 'SMA_50' in latest and latest['SMA_20'] > latest['SMA_50']:
            score += 0.1
        return {'score': score}
    def _sentiment_signal(self, data: pd.DataFrame) -> Dict:
        # Placeholder for sentiment analysis
        score = 0.0  # Extend with real sentiment
        return {'score': score}
    def _macro_signal(self) -> Dict:
        # Placeholder for macro signals
        score = 0.0  # Extend with real macro data
        return {'score': score}
    def _detect_regime(self, data: pd.DataFrame) -> str:
        # Simple regime detection (stub)
        returns = data['Close'].pct_change().dropna()
        if returns.mean() > 0.01:
            return 'bull'
        elif returns.mean() < -0.01:
            return 'bear'
        else:
            return 'sideways'
    def explain(self, data: pd.DataFrame, symbol: str) -> Dict:
        # Explain the latest signal (stub)
        if not self.signals:
            return {}
        last_signal = self.signals[-1]
        explanation = {
            'symbol': symbol,
            'regime': last_signal['regime'],
            'weights': self.config.get('weights', {}),
            'details': last_signal['details']
        }
        logger.info(f"[STRATEGY EXPLAIN] {explanation}")
        return explanation
    def report(self, symbol: str = None) -> None:
        print(f"\nSTRATEGY REPORT for {symbol if symbol else 'ALL'}:")
        signals = [s for s in self.signals if (symbol is None or s['symbol'] == symbol)]
        for s in signals[-5:]:
            print(f"  {s['timestamp']}: {s['signal']} (score: {s['score']:.2f}) regime: {s['regime']}")
    # RL hooks (stub)
    def train_rl_agent(self, *args, **kwargs):
        logger.info("[RL] RL agent training stub called.")
    def act_rl_agent(self, *args, **kwargs):
        logger.info("[RL] RL agent action stub called.")
    # Backtest with walk-forward, cross-validation (stub)
    def backtest(self, data: pd.DataFrame, symbol: str, walk_forward: bool = True, cv_folds: int = 5):
        logger.info(f"[BACKTEST] Backtest stub called for {symbol} walk_forward={walk_forward} cv_folds={cv_folds}") 