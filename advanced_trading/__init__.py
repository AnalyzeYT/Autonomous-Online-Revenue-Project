"""
Advanced Stock Trading System for Indian Markets
A comprehensive, professional-grade trading system with ML capabilities.
"""

from .data import DataCollector
from .model import AdvancedLSTMModel
from .strategy import TradingStrategy
from .portfolio import PortfolioManager
from .risk import RiskManager
from .dashboard import TradingDashboard
from .alerts import AlertSystem
from .backtest import BacktestEngine
from .main import TradingSystem, BatchProcessor, main, quick_analysis, run_backtest_demo
from .utils import setup_gpu_tpu

__version__ = "1.0.0"
__author__ = "Advanced Trading System Team"

__all__ = [
    'DataCollector',
    'AdvancedLSTMModel', 
    'TradingStrategy',
    'PortfolioManager',
    'RiskManager',
    'TradingDashboard',
    'AlertSystem',
    'BacktestEngine',
    'TradingSystem',
    'BatchProcessor',
    'main',
    'quick_analysis',
    'run_backtest_demo',
    'setup_gpu_tpu'
] 