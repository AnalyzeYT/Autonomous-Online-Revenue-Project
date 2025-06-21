"""
risk.py
200% Advanced Risk Management for the Advanced Stock Trading System.
Includes abstract base class, pluggable risk models, scenario analysis, Monte Carlo, risk attribution, and advanced reporting.
"""
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Any, Optional

# Set up logging for the module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Abstract base class for risk models
class BaseRiskModel(ABC):
    """
    Abstract base class for all risk models.
    """
    @abstractmethod
    def calculate(self, returns: np.ndarray, **kwargs) -> Dict[str, float]:
        pass
    @abstractmethod
    def report(self, returns: np.ndarray, **kwargs) -> None:
        pass

# Standard risk model (VaR, CVaR, volatility, max drawdown)
class StandardRiskModel(BaseRiskModel):
    def calculate(self, returns: np.ndarray, **kwargs) -> Dict[str, float]:
        var = np.percentile(returns, (1 - kwargs.get('confidence_level', 0.99)) * 100) if len(returns) > 1 else 0.0
        cvar = returns[returns <= -var].mean() if len(returns) > 1 and np.any(returns <= -var) else 0.0
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
        max_dd = self._calculate_max_drawdown_from_returns(returns)
        return {
            'VaR_99': abs(var),
            'CVaR_99': abs(cvar),
            'Volatility': volatility,
            'Max_Drawdown': max_dd
        }
    def report(self, returns: np.ndarray, **kwargs) -> None:
        metrics = self.calculate(returns, **kwargs)
        logger.info(f"[RISK REPORT] {metrics}")
        print("\nRISK SUMMARY:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    def _calculate_max_drawdown_from_returns(self, returns: np.ndarray) -> float:
        if len(returns) < 2:
            return 0.0
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return abs(drawdown.min())

# Monte Carlo scenario analysis risk model
class MonteCarloRiskModel(BaseRiskModel):
    def calculate(self, returns: np.ndarray, n_sim: int = 1000, horizon: int = 20, **kwargs) -> Dict[str, float]:
        if len(returns) < 2:
            return {'VaR_MC': 0.0, 'CVaR_MC': 0.0, 'Worst_Loss': 0.0}
        mu = np.mean(returns)
        sigma = np.std(returns)
        sim_results = np.zeros(n_sim)
        for i in range(n_sim):
            sim_path = np.random.normal(mu, sigma, horizon)
            sim_results[i] = np.prod(1 + sim_path) - 1
        var_mc = np.percentile(sim_results, 1)
        cvar_mc = sim_results[sim_results <= var_mc].mean() if np.any(sim_results <= var_mc) else 0.0
        worst_loss = np.min(sim_results)
        return {'VaR_MC': abs(var_mc), 'CVaR_MC': abs(cvar_mc), 'Worst_Loss': abs(worst_loss)}
    def report(self, returns: np.ndarray, **kwargs) -> None:
        metrics = self.calculate(returns, **kwargs)
        logger.info(f"[MC RISK REPORT] {metrics}")
        print("\nMONTE CARLO RISK SUMMARY:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

# Factor risk attribution model (stub)
class FactorRiskModel(BaseRiskModel):
    def calculate(self, returns: np.ndarray, factors: np.ndarray = None, **kwargs) -> Dict[str, float]:
        if factors is None or len(returns) != len(factors):
            logger.warning("[FACTOR] No valid factors provided for attribution.")
            return {'Factor_Contribution': 0.0}
        # Simple regression for factor attribution
        beta = np.cov(returns, factors)[0, 1] / np.var(factors)
        factor_contribution = beta * np.mean(factors)
        return {'Factor_Contribution': factor_contribution}
    def report(self, returns: np.ndarray, factors: np.ndarray = None, **kwargs) -> None:
        metrics = self.calculate(returns, factors, **kwargs)
        logger.info(f"[FACTOR RISK REPORT] {metrics}")
        print("\nFACTOR RISK SUMMARY:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

# Pluggable risk manager
class RiskManager:
    """
    200% Advanced Risk Manager: supports pluggable models, scenario analysis, attribution, config-driven.
    """
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.models: Dict[str, BaseRiskModel] = {
            'standard': StandardRiskModel(),
            'monte_carlo': MonteCarloRiskModel(),
            'factor': FactorRiskModel(),
        }
        self.active_models = self.config.get('active_models', ['standard'])
        self.initial_capital = self.config.get('initial_capital', 100000)
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.max_position_size = self.config.get('max_position_size', 0.1)
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.05)
        self.take_profit_pct = self.config.get('take_profit_pct', 0.15)
    def calculate_position_size(self, price: float, volatility: float = 0.02) -> int:
        win_rate = 0.6
        avg_win = 0.15
        avg_loss = 0.05
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))
        position_value = self.current_capital * kelly_fraction
        shares = int(position_value / price)
        return shares
    def check_risk_limits(self, symbol: str, price: float, shares: int) -> Tuple[bool, str]:
        position_value = price * shares
        position_pct = position_value / self.current_capital
        if position_pct > self.max_position_size:
            return False, f"Position size {position_pct:.2%} exceeds limit {self.max_position_size:.2%}"
        return True, "Risk check passed"
    def risk_report(self, returns: np.ndarray, factors: np.ndarray = None) -> Dict[str, Any]:
        report = {}
        for model_name in self.active_models:
            model = self.models.get(model_name)
            if model_name == 'factor' and factors is not None:
                metrics = model.calculate(returns, factors)
            else:
                metrics = model.calculate(returns)
            report[model_name] = metrics
        logger.info(f"[RISK MANAGER REPORT] {report}")
        return report
    def print_risk_summary(self, returns: np.ndarray, factors: np.ndarray = None) -> None:
        for model_name in self.active_models:
            model = self.models.get(model_name)
            if model_name == 'factor' and factors is not None:
                model.report(returns, factors)
            else:
                model.report(returns) 