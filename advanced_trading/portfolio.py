"""
portfolio.py
Portfolio management for the Advanced Stock Trading System.
"""
from typing import Dict, List, Optional
from datetime import datetime

class PortfolioManager:
    """
    Advanced portfolio management with optimization and performance tracking.
    """
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, int] = {}
        self.trades: List[Dict] = []
        self.performance_metrics: Dict = {}

    def execute_trade(self, symbol: str, action: str, shares: int, price: float, timestamp: datetime) -> Dict:
        """
        Execute buy/sell trades with comprehensive tracking.
        """
        trade_value = shares * price
        commission = max(trade_value * 0.001, 1)  # 0.1% commission, min $1
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': action,
            'shares': shares,
            'price': price,
            'value': trade_value,
            'commission': commission
        }
        if action.upper() == 'BUY':
            if self.cash >= trade_value + commission:
                self.cash -= (trade_value + commission)
                self.positions[symbol] = self.positions.get(symbol, 0) + shares
                trade['status'] = 'EXECUTED'
            else:
                trade['status'] = 'REJECTED - Insufficient funds'
        elif action.upper() == 'SELL':
            if self.positions.get(symbol, 0) >= shares:
                self.cash += (trade_value - commission)
                self.positions[symbol] -= shares
                if self.positions[symbol] == 0:
                    del self.positions[symbol]
                trade['status'] = 'EXECUTED'
            else:
                trade['status'] = 'REJECTED - Insufficient shares'
        self.trades.append(trade)
        return trade

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate current portfolio value.
        """
        portfolio_value = self.cash
        for symbol, shares in self.positions.items():
            if symbol in current_prices:
                portfolio_value += shares * current_prices[symbol]
        return portfolio_value

    def calculate_performance_metrics(self, current_prices: Dict[str, float]) -> Dict:
        """
        Calculate comprehensive performance metrics.
        """
        current_value = self.get_portfolio_value(current_prices)
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
            import numpy as np
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
        """
        Calculate maximum drawdown for the portfolio.
        """
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