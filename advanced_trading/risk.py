"""
risk.py
Risk management for the Advanced Stock Trading System.
"""
import numpy as np
from typing import Dict, Tuple

class RiskManager:
    """
    Professional-grade risk management system for trading simulation.
    Implements position sizing, stop-loss, take-profit, and VaR.
    """
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.max_position_size = 0.1  # 10% max position size
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.take_profit_pct = 0.15  # 15% take profit

    def calculate_position_size(self, price: float, volatility: float = 0.02) -> int:
        """
        Calculate optimal position size using Kelly Criterion.
        """
        win_rate = 0.6
        avg_win = 0.15
        avg_loss = 0.05
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))
        position_value = self.current_capital * kelly_fraction
        shares = int(position_value / price)
        return shares

    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.99) -> float:
        """
        Calculate Value at Risk (VaR) for the portfolio.
        """
        if len(returns) < 2:
            return 0.0
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return abs(var)

    def check_risk_limits(self, symbol: str, price: float, shares: int) -> Tuple[bool, str]:
        """
        Check if a trade meets risk management criteria.
        """
        position_value = price * shares
        position_pct = position_value / self.current_capital
        if position_pct > self.max_position_size:
            return False, f"Position size {position_pct:.2%} exceeds limit {self.max_position_size:.2%}"
        return True, "Risk check passed" 