"""
portfolio.py
200% Advanced Portfolio Management for the Advanced Stock Trading System.
Includes abstract base class, multi-asset/currency, advanced analytics, optimization stubs, and export/import.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import logging
import os
from abc import ABC, abstractmethod
import numpy as np

# Set up logging for the module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Abstract base class for portfolio managers
class BasePortfolioManager(ABC):
    """
    Abstract base class for all portfolio managers.
    """
    @abstractmethod
    def execute_trade(self, symbol: str, action: str, shares: int, price: float, timestamp: datetime, currency: str = 'INR', asset_type: str = 'stock') -> Dict:
        pass
    @abstractmethod
    def get_portfolio_value(self, current_prices: Dict[str, float], currency_rates: Dict[str, float] = None) -> float:
        pass
    @abstractmethod
    def calculate_performance_metrics(self, current_prices: Dict[str, float], currency_rates: Dict[str, float] = None) -> Dict:
        pass
    @abstractmethod
    def export_portfolio(self, filepath: str):
        pass
    @abstractmethod
    def import_portfolio(self, filepath: str):
        pass

# Advanced portfolio manager
class PortfolioManager(BasePortfolioManager):
    """
    200% Advanced Portfolio Manager: multi-asset, multi-currency, advanced analytics, optimization stubs, export/import.
    """
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.initial_capital = self.config.get('initial_capital', 100000)
        self.cash = self.initial_capital
        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol: {shares, currency, asset_type}
        self.trades: List[Dict] = []
        self.performance_metrics: Dict = {}
        self.currency_rates: Dict[str, float] = self.config.get('currency_rates', {'INR': 1.0})
    def execute_trade(self, symbol: str, action: str, shares: int, price: float, timestamp: datetime, currency: str = 'INR', asset_type: str = 'stock') -> Dict:
        trade_value = shares * price * self.currency_rates.get(currency, 1.0)
        commission = max(trade_value * 0.001, 1)
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': action,
            'shares': shares,
            'price': price,
            'currency': currency,
            'asset_type': asset_type,
            'value': trade_value,
            'commission': commission
        }
        if action.upper() == 'BUY':
            if self.cash >= trade_value + commission:
                self.cash -= (trade_value + commission)
                if symbol not in self.positions:
                    self.positions[symbol] = {'shares': 0, 'currency': currency, 'asset_type': asset_type}
                self.positions[symbol]['shares'] += shares
                trade['status'] = 'EXECUTED'
            else:
                trade['status'] = 'REJECTED - Insufficient funds'
        elif action.upper() == 'SELL':
            if symbol in self.positions and self.positions[symbol]['shares'] >= shares:
                self.cash += (trade_value - commission)
                self.positions[symbol]['shares'] -= shares
                if self.positions[symbol]['shares'] == 0:
                    del self.positions[symbol]
                trade['status'] = 'EXECUTED'
            else:
                trade['status'] = 'REJECTED - Insufficient shares'
        self.trades.append(trade)
        return trade
    def get_portfolio_value(self, current_prices: Dict[str, float], currency_rates: Dict[str, float] = None) -> float:
        currency_rates = currency_rates or self.currency_rates
        portfolio_value = self.cash
        for symbol, pos in self.positions.items():
            if symbol in current_prices:
                rate = currency_rates.get(pos['currency'], 1.0)
                portfolio_value += pos['shares'] * current_prices[symbol] * rate
        return portfolio_value
    def calculate_performance_metrics(self, current_prices: Dict[str, float], currency_rates: Dict[str, float] = None) -> Dict:
        current_value = self.get_portfolio_value(current_prices, currency_rates)
        total_return = (current_value - self.initial_capital) / self.initial_capital
        returns = []
        for trade in self.trades:
            if trade['status'] == 'EXECUTED' and trade['action'] == 'SELL':
                buy_trades = [t for t in self.trades if t['symbol'] == trade['symbol'] and t['action'] == 'BUY' and t['status'] == 'EXECUTED']
                if buy_trades:
                    buy_price = buy_trades[-1]['price']
                    trade_return = (trade['price'] - buy_price) / buy_price
                    returns.append(trade_return)
        if returns:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            max_drawdown = self.calculate_max_drawdown()
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        metrics = {
            'Total_Return': total_return,
            'Current_Value': current_value,
            'Cash': self.cash,
            'Positions': len(self.positions),
            'Total_Trades': len([t for t in self.trades if t['status'] == 'EXECUTED']),
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown
        }
        self.performance_metrics = metrics
        return metrics
    def calculate_max_drawdown(self) -> float:
        if len(self.trades) < 2:
            return 0.0
        values = [self.initial_capital]
        running_value = self.initial_capital
        for trade in self.trades:
            if trade['status'] == 'EXECUTED':
                if trade['action'] == 'BUY':
                    running_value -= trade['value'] + trade['commission']
                else:
                    running_value += trade['value'] - trade['commission']
                values.append(running_value)
        peak = values[0]
        max_dd = 0.0
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd
    def get_sector_allocation(self, sector_map: dict) -> dict:
        allocation = {sector: 0 for sector in sector_map}
        for symbol, pos in self.positions.items():
            for sector, symbols in sector_map.items():
                if symbol in symbols:
                    allocation[sector] += pos['shares']
        total = sum(allocation.values())
        if total > 0:
            allocation = {k: v / total for k, v in allocation.items()}
        return allocation
    def calculate_turnover(self) -> float:
        if not self.trades:
            return 0.0
        total_traded = sum(abs(t['value']) for t in self.trades if t['status'] == 'EXECUTED')
        avg_value = (self.initial_capital + self.get_portfolio_value({})) / 2
        return total_traded / avg_value if avg_value > 0 else 0.0
    def export_portfolio(self, filepath: str):
        state = {
            'cash': self.cash,
            'positions': self.positions,
            'trades': self.trades,
            'performance_metrics': self.performance_metrics
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        logger.info(f"[PORTFOLIO] Exported to {filepath}")
    def import_portfolio(self, filepath: str):
        if not os.path.exists(filepath):
            logger.error(f"[PORTFOLIO] File {filepath} does not exist.")
            return
        with open(filepath, 'r') as f:
            state = json.load(f)
        self.cash = state.get('cash', self.initial_capital)
        self.positions = state.get('positions', {})
        self.trades = state.get('trades', [])
        self.performance_metrics = state.get('performance_metrics', {})
        logger.info(f"[PORTFOLIO] Imported from {filepath}")
    # Portfolio optimization stubs
    def optimize_mean_variance(self, expected_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        # Placeholder for mean-variance optimization
        n = len(expected_returns)
        weights = np.ones(n) / n
        logger.info("[OPTIMIZATION] Mean-variance optimization stub called.")
        return weights
    def optimize_black_litterman(self, *args, **kwargs) -> np.ndarray:
        # Placeholder for Black-Litterman optimization
        logger.info("[OPTIMIZATION] Black-Litterman optimization stub called.")
        return np.array([])
    def optimize_risk_parity(self, *args, **kwargs) -> np.ndarray:
        # Placeholder for risk parity optimization
        logger.info("[OPTIMIZATION] Risk parity optimization stub called.")
        return np.array([]) 