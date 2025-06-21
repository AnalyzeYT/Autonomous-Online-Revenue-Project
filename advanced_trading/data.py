"""
data.py
Data ingestion and feature engineering for the Advanced Stock Trading System.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import talib
from typing import List, Dict, Optional, Tuple

class DataCollector:
    """
    Collects and processes stock data from multiple sources for Indian markets.
    Provides feature engineering for LSTM and other ML models.
    """
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()

    def get_indian_stocks_list(self) -> List[str]:
        """Returns a curated list of major Indian stock tickers (NSE)."""
        return [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS',
            'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'AXISBANK.NS', 'LT.NS',
            'HCLTECH.NS', 'WIPRO.NS', 'ULTRACEMCO.NS', 'TITAN.NS', 'NESTLEIND.NS'
        ]

    def fetch_stock_data(self, symbol: str, period: str = '2y', interval: str = '1d') -> Optional[Dict]:
        """
        Fetches historical stock data for a given symbol using yfinance.
        Returns a dict with 'data', 'info', and 'symbol' or None if unavailable.
        """
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval=interval)
            if data.empty:
                print(f"[WARN] No data found for {symbol}")
                return None
            info = stock.info
            return {'data': data, 'info': info, 'symbol': symbol}
        except Exception as e:
            print(f"[ERROR] Error fetching {symbol}: {e}")
            return None

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a comprehensive set of technical indicators to the dataframe.
        """
        data = df.copy()
        data['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)
        data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
        data['EMA_12'] = talib.EMA(data['Close'], timeperiod=12)
        data['EMA_26'] = talib.EMA(data['Close'], timeperiod=26)
        data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
        data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['Close'])
        data['STOCH_K'], data['STOCH_D'] = talib.STOCH(data['High'], data['Low'], data['Close'])
        data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['Close'])
        data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'])
        data['OBV'] = talib.OBV(data['Close'], data['Volume'])
        data['AD'] = talib.AD(data['High'], data['Low'], data['Close'], data['Volume'])
        data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'])
        data['CCI'] = talib.CCI(data['High'], data['Low'], data['Close'])
        data['DOJI'] = talib.CDLDOJI(data['Open'], data['High'], data['Low'], data['Close'])
        data['HAMMER'] = talib.CDLHAMMER(data['Open'], data['High'], data['Low'], data['Close'])
        data['Price_Change'] = data['Close'].pct_change()
        data['Volume_Change'] = data['Volume'].pct_change()
        data['High_Low_Ratio'] = (data['High'] - data['Low']) / data['Close']
        data['Open_Close_Ratio'] = (data['Open'] - data['Close']) / data['Close']
        return data.fillna(method='ffill').fillna(method='bfill')

    def prepare_lstm_data(
        self, data: pd.DataFrame, lookback: int = 60, forecast_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepares data for LSTM model training.
        Returns X, y, and the list of feature columns used.
        """
        feature_columns = [
            'Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'RSI', 'MACD', 'MACD_signal', 'STOCH_K', 'STOCH_D',
            'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'OBV', 'AD',
            'ADX', 'CCI', 'Price_Change', 'Volume_Change',
            'High_Low_Ratio', 'Open_Close_Ratio'
        ]
        available_features = [col for col in feature_columns if col in data.columns]
        feature_data = data[available_features].values
        scaled_features = self.feature_scaler.fit_transform(feature_data)
        target_data = data['Close'].values.reshape(-1, 1)
        scaled_target = self.scaler.fit_transform(target_data)
        X, y = [], []
        for i in range(lookback, len(scaled_features) - forecast_horizon + 1):
            X.append(scaled_features[i-lookback:i])
            y.append(scaled_target[i:i+forecast_horizon])
        return np.array(X), np.array(y), available_features 