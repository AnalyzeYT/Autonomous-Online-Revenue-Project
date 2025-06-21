"""
main.py
Colab-ready main script tying all advanced modules together for the Advanced Stock Trading System.
- Device/strategy detection (TPU/GPU/CPU)
- Config loading
- Instantiates all advanced modules
- Demonstrates full workflow: data, model, risk, portfolio, strategy, dashboard, alerts
- Designed for Colab, modular, and extensible
"""
import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Import advanced modules
from .data import DataCollector
from .model import get_model, ModelRegistry
from .risk import RiskManager
from .portfolio import PortfolioManager
from .strategy import TradingStrategy
from .dashboard import TradingDashboard
from .alerts import AlertSystem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Device/strategy detection (TPU/GPU/CPU)
import tensorflow as tf

def get_strategy():
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        print("✅ TPU detected and initialized.")
    except Exception:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            strategy = tf.distribute.MirroredStrategy()
            print(f"✅ GPU detected: {len(gpus)} device(s).")
        else:
            strategy = tf.distribute.get_strategy()
            print("⚠️ Using CPU only.")
    return strategy

strategy = get_strategy()

# 2. Config (can be loaded from file or defined here)
config = {
    'initial_capital': 100000,
    'feature_engineering': {
        'sma': True, 'ema': True, 'rsi': True, 'macd': True, 'stoch': True, 'bbands': True,
        'atr': True, 'obv': True, 'ad': True, 'adx': True, 'cci': True
    },
    'default_source': 'yahoo',
    'max_position_size': 0.1,
    'stop_loss_pct': 0.05,
    'take_profit_pct': 0.15,
    'currency_rates': {'INR': 1.0},
    'weights': {'ml': 0.5, 'tech': 0.2, 'sentiment': 0.2, 'macro': 0.1},
}

# 3. Instantiate modules
print("\n🚀 Initializing advanced trading system modules...")
data_collector = DataCollector(config)
portfolio = PortfolioManager(config)
risk_manager = RiskManager(config)
dashboard = TradingDashboard(config)
alerts = AlertSystem(config)

# Add default dashboard panels and alert channels
dashboard.add_default_panels()
alerts.add_default_channels()

# 4. Data fetch and feature engineering
symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
print(f"\n📊 Fetching data for: {symbols}")
data_dict = {}
for symbol in symbols:
    stock_data = data_collector.fetch_stock_data(symbol, period='2y', interval='1d')
    if stock_data:
        processed = data_collector.calculate_technical_indicators(stock_data['data'])
        data_dict[symbol] = processed
    else:
        print(f"❌ Could not fetch data for {symbol}")

# 5. Model training and evaluation (for one symbol)
symbol = symbols[0]
print(f"\n🤖 Training model for {symbol}...")
processed_data = data_dict[symbol]
X, y, features = data_collector.prepare_lstm_data(processed_data)
train_size = int(len(X) * 0.8)
val_size = int(len(X) * 0.1)
X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]
input_shape = (X.shape[1], X.shape[2])

with strategy.scope():
    model = get_model('lstm', input_shape, config)
    model.build_model()
    history = model.train(X_train, y_train, X_val, y_val, config)
    metrics, predictions, actual = model.evaluate(X_test, y_test, data_collector.engineer.scaler)

print(f"\n✅ Model trained for {symbol}. RMSE: {metrics['RMSE']:.2f}, Directional Accuracy: {metrics['Directional_Accuracy']:.1f}%")

# 6. Risk, portfolio, and strategy
returns = processed_data['Close'].pct_change().dropna().values
risk_manager.print_risk_summary(returns)

# Simulate a trade
current_price = processed_data['Close'].iloc[-1]
shares = risk_manager.calculate_position_size(current_price)
trade = portfolio.execute_trade(symbol, 'BUY', shares, current_price, datetime.now())
print(f"\n💼 Executed trade: {trade}")

# 7. Strategy and alerts
strategy = TradingStrategy(model, data_collector.engineer.scaler, data_collector.engineer.feature_scaler, config)
signal = strategy.generate_signals(processed_data, symbol)
print(f"\n🎯 Strategy signal: {signal}")
if signal and signal['signal'] in ['BUY', 'SELL']:
    alerts.send_alert({
        'type': 'TRADE_SIGNAL',
        'symbol': symbol,
        'message': f"{signal['signal']} with score {signal['score']:.2f}",
        'priority': 'HIGH' if abs(signal['score']) > 0.5 else 'MEDIUM',
        'timestamp': datetime.now()
    })

# 8. Dashboard rendering and export
dashboard_data = {
    'prices': processed_data,
    'analytics': metrics
}
rendered = dashboard.render(dashboard_data)
html = dashboard.export(export_type='html', filepath='dashboard.html')
print("\n📊 Dashboard exported to dashboard.html")

# 9. Alert reporting
alerts.report()

# 10. Portfolio performance
current_prices = {symbol: processed_data['Close'].iloc[-1] for symbol in symbols}
metrics = portfolio.calculate_performance_metrics(current_prices)
print(f"\n💼 Portfolio metrics: {metrics}")

print("\n🎉 Advanced Trading System Colab workflow complete!") 