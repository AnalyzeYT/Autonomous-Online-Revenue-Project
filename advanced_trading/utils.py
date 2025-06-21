"""
utils.py
Utility functions for the Advanced Stock Trading System.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

def setup_gpu_tpu():
    """
    Configure GPU/TPU for optimal performance in TensorFlow.
    Returns the appropriate distribution strategy.
    """
    import tensorflow as tf
    
    print("🔧 Setting up hardware acceleration...")
    
    # Check for GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"✅ GPU Found: {len(physical_devices)} device(s)")
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled")
        except RuntimeError as e:
            print(f"⚠️ GPU setup warning: {e}")
    
    # Check for TPU
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

def calculate_technical_indicators_manual(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators manually without talib dependency.
    Useful for environments where talib is not available.
    """
    data = df.copy()
    
    # Simple Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_hist'] = data['MACD'] - data['MACD_signal']
    
    # Bollinger Bands
    data['BB_middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
    data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
    
    # Stochastic Oscillator
    low_min = data['Low'].rolling(window=14).min()
    high_max = data['High'].rolling(window=14).max()
    data['STOCH_K'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    data['STOCH_D'] = data['STOCH_K'].rolling(window=3).mean()
    
    # Average True Range (ATR)
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    data['ATR'] = true_range.rolling(window=14).mean()
    
    # On-Balance Volume (OBV)
    data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    
    # Price and Volume Changes
    data['Price_Change'] = data['Close'].pct_change()
    data['Volume_Change'] = data['Volume'].pct_change()
    data['High_Low_Ratio'] = (data['High'] - data['Low']) / data['Close']
    data['Open_Close_Ratio'] = (data['Open'] - data['Close']) / data['Close']
    
    return data.fillna(method='ffill').fillna(method='bfill')

def validate_data_quality(df: pd.DataFrame) -> Dict:
    """
    Validate data quality and return quality metrics.
    """
    quality_report = {
        'total_rows': len(df),
        'missing_values': {},
        'duplicates': len(df.duplicated()),
        'date_range': {},
        'price_anomalies': 0,
        'volume_anomalies': 0
    }
    
    # Check missing values
    for column in df.columns:
        missing_count = df[column].isnull().sum()
        if missing_count > 0:
            quality_report['missing_values'][column] = missing_count
    
    # Check date range
    if 'Date' in df.columns or df.index.dtype == 'datetime64[ns]':
        dates = df.index if df.index.dtype == 'datetime64[ns]' else df['Date']
        quality_report['date_range'] = {
            'start': dates.min(),
            'end': dates.max(),
            'total_days': (dates.max() - dates.min()).days
        }
    
    # Check for price anomalies (negative or zero prices)
    if 'Close' in df.columns:
        price_anomalies = len(df[df['Close'] <= 0])
        quality_report['price_anomalies'] = price_anomalies
    
    # Check for volume anomalies
    if 'Volume' in df.columns:
        volume_anomalies = len(df[df['Volume'] < 0])
        quality_report['volume_anomalies'] = volume_anomalies
    
    return quality_report

def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series
        method: 'simple' or 'log'
    
    Returns:
        Returns series
    """
    if method == 'simple':
        return prices.pct_change()
    elif method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError("Method must be 'simple' or 'log'")

def calculate_rolling_metrics(returns: pd.Series, window: int = 252) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.
    
    Args:
        returns: Returns series
        window: Rolling window size
    
    Returns:
        DataFrame with rolling metrics
    """
    rolling_metrics = pd.DataFrame()
    
    # Rolling returns
    rolling_metrics['rolling_return'] = returns.rolling(window=window).mean() * 252
    
    # Rolling volatility
    rolling_metrics['rolling_volatility'] = returns.rolling(window=window).std() * np.sqrt(252)
    
    # Rolling Sharpe ratio (assuming risk-free rate of 0.02)
    risk_free_rate = 0.02
    rolling_metrics['rolling_sharpe'] = (
        (rolling_metrics['rolling_return'] - risk_free_rate) / 
        rolling_metrics['rolling_volatility']
    )
    
    # Rolling maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.rolling(window=window).max()
    rolling_metrics['rolling_drawdown'] = (cumulative_returns - rolling_max) / rolling_max
    
    return rolling_metrics

def detect_outliers(data: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers in a data series.
    
    Args:
        data: Input data series
        method: 'iqr' (Interquartile Range) or 'zscore'
        threshold: Threshold for outlier detection
    
    Returns:
        Boolean series indicating outliers
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (data < lower_bound) | (data > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")

def calculate_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix for multiple assets.
    
    Args:
        returns_df: DataFrame with returns for multiple assets
    
    Returns:
        Correlation matrix
    """
    return returns_df.corr()

def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calculate beta for an asset relative to the market.
    
    Args:
        asset_returns: Asset returns series
        market_returns: Market returns series
    
    Returns:
        Beta value
    """
    # Align the series
    aligned_data = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if len(aligned_data) < 2:
        return 0.0
    
    asset_ret = aligned_data.iloc[:, 0]
    market_ret = aligned_data.iloc[:, 1]
    
    # Calculate beta
    covariance = np.cov(asset_ret, market_ret)[0, 1]
    market_variance = np.var(market_ret)
    
    return covariance / market_variance if market_variance > 0 else 0.0

def calculate_alpha(asset_returns: pd.Series, market_returns: pd.Series, 
                   risk_free_rate: float = 0.02) -> float:
    """
    Calculate alpha for an asset.
    
    Args:
        asset_returns: Asset returns series
        market_returns: Market returns series
        risk_free_rate: Risk-free rate (annual)
    
    Returns:
        Alpha value (annualized)
    """
    beta = calculate_beta(asset_returns, market_returns)
    
    # Align the series
    aligned_data = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if len(aligned_data) < 2:
        return 0.0
    
    asset_ret = aligned_data.iloc[:, 0]
    market_ret = aligned_data.iloc[:, 1]
    
    # Calculate alpha
    asset_mean = np.mean(asset_ret)
    market_mean = np.mean(market_ret)
    rf_daily = risk_free_rate / 252
    
    alpha = (asset_mean - rf_daily) - beta * (market_mean - rf_daily)
    return alpha * 252  # Annualize

def format_currency(amount: float, currency: str = '₹') -> str:
    """
    Format currency amounts for display.
    
    Args:
        amount: Amount to format
        currency: Currency symbol
    
    Returns:
        Formatted currency string
    """
    if abs(amount) >= 1e9:
        return f"{currency}{amount/1e9:.2f}B"
    elif abs(amount) >= 1e6:
        return f"{currency}{amount/1e6:.2f}M"
    elif abs(amount) >= 1e3:
        return f"{currency}{amount/1e3:.2f}K"
    else:
        return f"{currency}{amount:.2f}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format percentage values for display.
    
    Args:
        value: Value to format as percentage
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    return f"{value:.{decimals}%}"

def get_market_hours() -> Dict:
    """
    Get Indian market trading hours.
    
    Returns:
        Dictionary with market hours information
    """
    return {
        'pre_market': '09:00-09:08',
        'regular_trading': '09:15-15:30',
        'post_market': '15:40-16:00',
        'timezone': 'IST',
        'days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    }

def is_market_open() -> bool:
    """
    Check if Indian market is currently open.
    
    Returns:
        True if market is open, False otherwise
    """
    now = datetime.now()
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Check if it's within trading hours (simplified)
    current_time = now.time()
    market_start = datetime.strptime('09:15', '%H:%M').time()
    market_end = datetime.strptime('15:30', '%H:%M').time()
    
    return market_start <= current_time <= market_end

def calculate_position_sizing_kelly(win_rate: float, avg_win: float, 
                                  avg_loss: float) -> float:
    """
    Calculate optimal position size using Kelly Criterion.
    
    Args:
        win_rate: Probability of winning
        avg_win: Average win amount
        avg_loss: Average loss amount
    
    Returns:
        Optimal fraction of capital to risk
    """
    if avg_loss == 0:
        return 0.0
    
    kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    return max(0, min(kelly_fraction, 0.25))  # Cap at 25%

def calculate_var_historical(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk using historical simulation.
    
    Args:
        returns: Returns series
        confidence_level: Confidence level (e.g., 0.95 for 95%)
    
    Returns:
        VaR value
    """
    if len(returns) < 2:
        return 0.0
    
    var = np.percentile(returns, (1 - confidence_level) * 100)
    return abs(var)

def calculate_expected_shortfall(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Expected Shortfall (Conditional VaR).
    
    Args:
        returns: Returns series
        confidence_level: Confidence level
    
    Returns:
        Expected Shortfall value
    """
    if len(returns) < 2:
        return 0.0
    
    var = calculate_var_historical(returns, confidence_level)
    tail_returns = returns[returns <= -var]
    
    if len(tail_returns) == 0:
        return 0.0
    
    return abs(tail_returns.mean()) 