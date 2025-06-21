"""
data.py
200% Advanced Data Ingestion and Feature Engineering for the Advanced Stock Trading System.
Includes pluggable data sources, async fetching, caching, feature store hooks, anomaly detection, and config-driven pipelines.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import talib
import asyncio
import concurrent.futures
import logging
import os
import hashlib
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any, Callable, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from joblib import Memory

# Set up logging for the module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up a cache directory for data versioning/caching
CACHE_DIR = os.environ.get("TRADING_DATA_CACHE", ".cache/data")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
memory = Memory(CACHE_DIR, verbose=0)

# Abstract base class for all data sources
class BaseDataSource(ABC):
    """
    Abstract base class for all data sources (API, CSV, DB, cloud, etc.)
    """
    @abstractmethod
    def fetch(self, symbol: str, **kwargs) -> Optional[pd.DataFrame]:
        pass

# Concrete implementation: Yahoo Finance API
class YahooFinanceDataSource(BaseDataSource):
    def fetch(self, symbol: str, period: str = '2y', interval: str = '1d', **kwargs) -> Optional[pd.DataFrame]:
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval=interval)
            if data.empty:
                logger.warning(f"[YF] No data for {symbol}")
                return None
            return data
        except Exception as e:
            logger.error(f"[YF] Error fetching {symbol}: {e}")
            return None

# Example: CSV Data Source (for backtesting/local data)
class CSVDataSource(BaseDataSource):
    def __init__(self, base_path: str):
        self.base_path = base_path
    def fetch(self, symbol: str, **kwargs) -> Optional[pd.DataFrame]:
        path = os.path.join(self.base_path, f"{symbol}.csv")
        if not os.path.exists(path):
            logger.warning(f"[CSV] File not found: {path}")
            return None
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            return df
        except Exception as e:
            logger.error(f"[CSV] Error reading {path}: {e}")
            return None

# Async/parallel data fetching utility
async def fetch_all_async(symbols: List[str], fetch_fn: Callable[[str], Optional[pd.DataFrame]], max_workers: int = 8) -> Dict[str, Optional[pd.DataFrame]]:
    loop = asyncio.get_event_loop()
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [loop.run_in_executor(executor, fetch_fn, symbol) for symbol in symbols]
        for symbol, task in zip(symbols, tasks):
            try:
                results[symbol] = await task
            except Exception as e:
                logger.error(f"[ASYNC] Error fetching {symbol}: {e}")
                results[symbol] = None
    return results

# Data versioning/caching utility
@memory.cache
def cached_fetch(symbol: str, source_name: str, params: dict) -> Optional[pd.DataFrame]:
    # Use a hash of params for versioning
    key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
    logger.info(f"[CACHE] Fetching {symbol} from {source_name} with key {key}")
    if source_name == 'yahoo':
        return YahooFinanceDataSource().fetch(symbol, **params)
    # Add more sources as needed
    return None

# Feature store integration (stub)
class FeatureStore:
    """
    Stub for feature store integration (e.g., Feast, custom, etc.)
    """
    def __init__(self):
        self.store = {}
    def get_features(self, symbol: str) -> Optional[pd.DataFrame]:
        return self.store.get(symbol)
    def save_features(self, symbol: str, features: pd.DataFrame):
        self.store[symbol] = features
        logger.info(f"[FEATURE STORE] Saved features for {symbol}")

# Data quality and anomaly detection utilities
class DataQualityChecker:
    @staticmethod
    def check_nan(df: pd.DataFrame) -> bool:
        nans = df.isnull().sum().sum()
        if nans > 0:
            logger.warning(f"[DQ] DataFrame contains {nans} NaN values.")
            return False
        return True
    @staticmethod
    def check_duplicates(df: pd.DataFrame) -> bool:
        if df.index.duplicated().any():
            logger.warning("[DQ] DataFrame contains duplicate indices.")
            return False
        return True
    @staticmethod
    def detect_anomalies(df: pd.DataFrame, z_thresh: float = 4.0) -> pd.DataFrame:
        # Simple z-score anomaly detection on 'Close'
        if 'Close' not in df.columns:
            return df
        z = (df['Close'] - df['Close'].mean()) / df['Close'].std()
        anomalies = df[np.abs(z) > z_thresh]
        if not anomalies.empty:
            logger.warning(f"[DQ] Detected {len(anomalies)} anomalies in 'Close'.")
        return anomalies

# Config-driven, extensible feature engineering pipeline
class FeatureEngineer:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        self.custom_funcs: List[Callable[[pd.DataFrame], pd.DataFrame]] = []
    def add_custom_feature(self, func: Callable[[pd.DataFrame], pd.DataFrame]):
        self.custom_funcs.append(func)
    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        # Standard indicators
        if self.config.get('sma', True):
            data['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)
            data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
        if self.config.get('ema', True):
            data['EMA_12'] = talib.EMA(data['Close'], timeperiod=12)
            data['EMA_26'] = talib.EMA(data['Close'], timeperiod=26)
        if self.config.get('rsi', True):
            data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
        if self.config.get('macd', True):
            data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['Close'])
        if self.config.get('stoch', True):
            data['STOCH_K'], data['STOCH_D'] = talib.STOCH(data['High'], data['Low'], data['Close'])
        if self.config.get('bbands', True):
            data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['Close'])
        if self.config.get('atr', True):
            data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'])
        if self.config.get('obv', True):
            data['OBV'] = talib.OBV(data['Close'], data['Volume'])
        if self.config.get('ad', True):
            data['AD'] = talib.AD(data['High'], data['Low'], data['Close'], data['Volume'])
        if self.config.get('adx', True):
            data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'])
        if self.config.get('cci', True):
            data['CCI'] = talib.CCI(data['High'], data['Low'], data['Close'])
        # Custom indicators
        data['Price_Change'] = data['Close'].pct_change()
        data['Volume_Change'] = data['Volume'].pct_change()
        data['High_Low_Ratio'] = (data['High'] - data['Low']) / data['Close']
        data['Open_Close_Ratio'] = (data['Open'] - data['Close']) / data['Close']
        # Apply user custom features
        for func in self.custom_funcs:
            data = func(data)
        # Fill missing
        data = data.fillna(method='ffill').fillna(method='bfill')
        return data

# Main advanced data collector
class DataCollector:
    """
    200% Advanced Data Collector: supports pluggable sources, async, caching, feature store, anomaly detection, config-driven pipelines.
    """
    def __init__(self, config: dict = None, feature_store: FeatureStore = None):
        self.config = config or {}
        self.feature_store = feature_store or FeatureStore()
        self.engineer = FeatureEngineer(self.config.get('feature_engineering', {}))
        self.sources: Dict[str, BaseDataSource] = {
            'yahoo': YahooFinanceDataSource(),
            # Add more sources as needed
        }
        self.default_source = self.config.get('default_source', 'yahoo')
    def fetch_stock_data(self, symbol: str, period: str = '2y', interval: str = '1d', source: str = None) -> Optional[Dict]:
        """
        Fetches stock data from the configured or specified source, with caching and quality checks.
        """
        src = source or self.default_source
        params = {'period': period, 'interval': interval}
        df = cached_fetch(symbol, src, params)
        if df is None:
            logger.error(f"[DATA] No data for {symbol} from {src}")
            return None
        # Data quality checks
        DataQualityChecker.check_nan(df)
        DataQualityChecker.check_duplicates(df)
        DataQualityChecker.detect_anomalies(df)
        return {'data': df, 'symbol': symbol, 'source': src}
    async def fetch_multiple_stocks_async(self, symbols: List[str], period: str = '2y', interval: str = '1d', source: str = None) -> Dict[str, Optional[Dict]]:
        """
        Async fetch for multiple stocks.
        """
        src = source or self.default_source
        fetch_fn = lambda symbol: self.fetch_stock_data(symbol, period, interval, src)
        results = await fetch_all_async(symbols, fetch_fn)
        return {k: v for k, v in results.items() if v is not None}
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the advanced, config-driven feature engineering pipeline.
        """
        return self.engineer.engineer(df)
    def prepare_lstm_data(self, data: pd.DataFrame, lookback: int = 60, forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepares data for LSTM model training.
        Returns X, y, and the list of feature columns used.
        """
        feature_columns = [col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        feature_data = data[feature_columns].values
        scaled_features = self.engineer.feature_scaler.fit_transform(feature_data)
        target_data = data['Close'].values.reshape(-1, 1)
        scaled_target = self.engineer.scaler.fit_transform(target_data)
        X, y = [], []
        for i in range(lookback, len(scaled_features) - forecast_horizon + 1):
            X.append(scaled_features[i-lookback:i])
            y.append(scaled_target[i:i+forecast_horizon])
        return np.array(X), np.array(y), feature_columns
    def save_features_to_store(self, symbol: str, features: pd.DataFrame):
        self.feature_store.save_features(symbol, features)
    def get_features_from_store(self, symbol: str) -> Optional[pd.DataFrame]:
        return self.feature_store.get_features(symbol)
    def get_indian_stocks_list(self) -> List[str]:
        return [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS',
            'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'AXISBANK.NS', 'LT.NS',
            'HCLTECH.NS', 'WIPRO.NS', 'ULTRACEMCO.NS', 'TITAN.NS', 'NESTLEIND.NS'
        ]
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'head': df.head(2).to_dict(),
            'tail': df.tail(2).to_dict()
        }
        return summary
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        for col in ['Close', 'Volume']:
            if col in data.columns:
                z = (data[col] - data[col].mean()) / data[col].std()
                data = data[(z > -3) & (z < 3)]
        data = data.fillna(method='ffill').fillna(method='bfill')
        return data 