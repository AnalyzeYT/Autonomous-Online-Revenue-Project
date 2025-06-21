# Advanced Stock Trading System for Indian Markets
# Optimized for Google Colab with GPU/TPU Support

# ===== INSTALLATION AND SETUP =====
# (In Colab, use !pip install ... as needed)

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json
import talib
import requests
from textblob import TextBlob

def setup_gpu_tpu():
    """Configure GPU/TPU for optimal performance"""
    print("🔧 Setting up hardware acceleration...")
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"✅ GPU Found: {len(physical_devices)} device(s)")
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled")
        except RuntimeError as e:
            print(f"⚠️ GPU setup warning: {e}")
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("✅ TPU initialized successfully")
        return strategy
    except ValueError:
        if physical_devices:
            strategy = tf.distribute.MirroredStrategy()
            print("✅ GPU strategy initialized")
        else:
            strategy = tf.distribute.get_strategy()
            print("⚠️ Using CPU - consider enabling GPU in Colab")
        return strategy

# Initialize hardware
strategy = setup_gpu_tpu()

class DataCollector:
    """Advanced data collection with multiple sources and real-time capabilities"""
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
    def get_indian_stocks_list(self):
        return [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS',
            'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'AXISBANK.NS', 'LT.NS',
            'HCLTECH.NS', 'WIPRO.NS', 'ULTRACEMCO.NS', 'TITAN.NS', 'NESTLEIND.NS'
        ]
    def fetch_stock_data(self, symbol, period='2y', interval='1d'):
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval=interval)
            if data.empty:
                print(f"⚠️ No data found for {symbol}")
                return None
            info = stock.info
            return {'data': data, 'info': info, 'symbol': symbol}
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
            return None
    def calculate_technical_indicators(self, df):
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
    def prepare_lstm_data(self, data, lookback=60, forecast_horizon=1):
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

# ... (rest of the code will be added in subsequent edits) ... 