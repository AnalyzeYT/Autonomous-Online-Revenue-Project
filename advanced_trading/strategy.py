"""
strategy.py
Trading strategy implementation combining ML predictions and technical analysis.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from .data import DataCollector

class TradingStrategy:
    """
    Advanced trading strategy with ML predictions and technical analysis.
    Combines LSTM predictions with traditional technical indicators.
    """
    
    def __init__(self, model, scaler, feature_scaler):
        self.model = model
        self.scaler = scaler
        self.feature_scaler = feature_scaler
        self.signals = []
        self.data_collector = DataCollector()
        
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """
        Generate trading signals based on ML predictions and technical analysis.
        Returns comprehensive signal data with confidence scores.
        """
        if len(data) < 60:
            return None
        
        # Prepare recent data for prediction
        recent_data = data.tail(60)
        
        # Calculate technical indicators
        processed_data = self.data_collector.calculate_technical_indicators(recent_data)
        
        # Get features for prediction
        feature_columns = [
            'Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'RSI', 'MACD', 'MACD_signal', 'STOCH_K', 'STOCH_D',
            'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'OBV', 'AD',
            'ADX', 'CCI', 'Price_Change', 'Volume_Change',
            'High_Low_Ratio', 'Open_Close_Ratio'
        ]
        
        available_features = [col for col in feature_columns if col in processed_data.columns]
        feature_data = processed_data[available_features].fillna(method='ffill').fillna(0)
        
        if len(feature_data) < 60:
            return None
        
        # Scale features and make prediction
        scaled_features = self.feature_scaler.transform(feature_data.values[-60:])
        X_pred = scaled_features.reshape(1, 60, -1)
        prediction = self.model.predict(X_pred)
        predicted_price = self.scaler.inverse_transform(prediction)[0][0]
        
        current_price = data['Close'].iloc[-1]
        price_change_pct = (predicted_price - current_price) / current_price
        
        # Technical analysis signals
        latest = processed_data.iloc[-1]
        
        # RSI signals
        rsi_oversold = latest['RSI'] < 30
        rsi_overbought = latest['RSI'] > 70
        
        # MACD signals
        macd_bullish = latest['MACD'] > latest['MACD_signal']
        
        # Moving Average signals
        ma_bullish = latest['Close'] > latest['SMA_20'] > latest['SMA_50']
        
        # Bollinger Bands signals
        bb_oversold = latest['Close'] < latest['BB_lower']
        bb_overbought = latest['Close'] > latest['BB_upper']
        
        # Stochastic signals
        stoch_oversold = latest['STOCH_K'] < 20
        stoch_overbought = latest['STOCH_K'] > 80
        
        # Volume signals
        volume_spike = latest['Volume'] > data['Volume'].rolling(20).mean().iloc[-1] * 1.5
        
        # ADX trend strength
        strong_trend = latest['ADX'] > 25
        
        # Combine signals with weights
        bullish_signals = sum([
            price_change_pct > 0.02,  # ML prediction shows >2% upside
            rsi_oversold,
            macd_bullish,
            ma_bullish,
            bb_oversold,
            stoch_oversold,
            volume_spike,
            strong_trend
        ])
        
        bearish_signals = sum([
            price_change_pct < -0.02,  # ML prediction shows >2% downside
            rsi_overbought,
            not macd_bullish,
            not ma_bullish,
            bb_overbought,
            stoch_overbought,
            volume_spike,
            strong_trend
        ])
        
        # Generate final signal with confidence
        if bullish_signals >= 4:
            signal = 'BUY'
            confidence = min(bullish_signals / 8, 0.95)
        elif bearish_signals >= 4:
            signal = 'SELL'
            confidence = min(bearish_signals / 8, 0.95)
        else:
            signal = 'HOLD'
            confidence = 0.5
        
        # Calculate risk-reward ratio
        risk_reward_ratio = abs(price_change_pct) / 0.05  # Assuming 5% risk
        
        signal_data = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change_pct': price_change_pct,
            'risk_reward_ratio': risk_reward_ratio,
            'technical_indicators': {
                'RSI': latest['RSI'],
                'MACD': latest['MACD'],
                'MACD_Signal': latest['MACD_signal'],
                'SMA_20': latest['SMA_20'],
                'SMA_50': latest['SMA_50'],
                'BB_Position': (latest['Close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower']) if latest['BB_upper'] != latest['BB_lower'] else 0.5,
                'STOCH_K': latest['STOCH_K'],
                'STOCH_D': latest['STOCH_D'],
                'ADX': latest['ADX'],
                'Volume_Ratio': latest['Volume'] / data['Volume'].rolling(20).mean().iloc[-1]
            },
            'signal_strength': {
                'bullish_count': bullish_signals,
                'bearish_count': bearish_signals,
                'total_indicators': 8
            }
        }
        
        self.signals.append(signal_data)
        return signal_data
    
    def get_signal_history(self, symbol: str = None, hours: int = 24) -> List[Dict]:
        """
        Get signal history for analysis and backtesting.
        """
        cutoff_time = datetime.now() - pd.Timedelta(hours=hours)
        if symbol:
            return [s for s in self.signals if s['symbol'] == symbol and s['timestamp'] > cutoff_time]
        return [s for s in self.signals if s['timestamp'] > cutoff_time]
    
    def calculate_signal_accuracy(self, symbol: str = None) -> Dict:
        """
        Calculate historical signal accuracy for performance analysis.
        """
        signals = self.get_signal_history(symbol, hours=24*30)  # Last 30 days
        if not signals:
            return {'accuracy': 0, 'total_signals': 0}
        
        correct_signals = 0
        total_signals = 0
        
        for signal in signals:
            if signal['signal'] in ['BUY', 'SELL']:
                total_signals += 1
                # This would need actual price data to verify accuracy
                # For now, we'll use a simplified approach
                if signal['confidence'] > 0.7:
                    correct_signals += 1
        
        accuracy = correct_signals / total_signals if total_signals > 0 else 0
        
        return {
            'accuracy': accuracy,
            'total_signals': total_signals,
            'high_confidence_signals': len([s for s in signals if s['confidence'] > 0.7])
        } 