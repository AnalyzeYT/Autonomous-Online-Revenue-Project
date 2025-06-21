"""
backtest.py
Comprehensive backtesting engine for trading strategies.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from .portfolio import PortfolioManager
from .risk import RiskManager

class BacktestEngine:
    """
    Comprehensive backtesting engine with advanced metrics and analysis.
    Supports multiple strategies, risk management, and performance evaluation.
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.portfolio = PortfolioManager(initial_capital)
        self.risk_manager = RiskManager(initial_capital)
        self.results = {}
        self.daily_returns = []
        self.equity_curve = []
        
    def run_backtest(self, strategy, data_dict: Dict[str, pd.DataFrame], 
                    start_date: datetime, end_date: datetime) -> Dict:
        """
        Run comprehensive backtest with detailed performance tracking.
        """
        print("🔄 Starting backtest...")
        
        # Filter data by date range
        filtered_data = {}
        for symbol, data in data_dict.items():
            mask = (data.index >= start_date) & (data.index <= end_date)
            filtered_data[symbol] = data.loc[mask]
        
        # Get all unique dates
        all_dates = set()
        for data in filtered_data.values():
            all_dates.update(data.index)
        all_dates = sorted(list(all_dates))
        
        # Initialize tracking
        self.daily_returns = []
        self.equity_curve = []
        daily_portfolio_values = []
        
        # Run backtest day by day
        for i, current_date in enumerate(all_dates):
            current_prices = {}
            
            # Get current prices for all symbols
            for symbol, data in filtered_data.items():
                if current_date in data.index:
                    current_prices[symbol] = data.loc[current_date, 'Close']
            
            # Generate signals for each symbol
            for symbol, data in filtered_data.items():
                if current_date not in data.index:
                    continue
                
                # Get historical data up to current date
                historical_data = data[data.index <= current_date]
                
                if len(historical_data) < 60:  # Need minimum data for signals
                    continue
                
                # Generate signal
                signal_data = strategy.generate_signals(historical_data, symbol)
                
                if signal_data is None:
                    continue
                
                current_price = signal_data['current_price']
                signal = signal_data['signal']
                confidence = signal_data['confidence']
                
                # Execute trades based on signals
                if signal == 'BUY' and confidence > 0.7:
                    shares = self.risk_manager.calculate_position_size(current_price)
                    if shares > 0:
                        risk_ok, risk_msg = self.risk_manager.check_risk_limits(symbol, current_price, shares)
                        if risk_ok:
                            self.portfolio.execute_trade(symbol, 'BUY', shares, current_price, current_date)
                
                elif signal == 'SELL' and confidence > 0.7:
                    current_position = self.portfolio.positions.get(symbol, 0)
                    if current_position > 0:
                        shares_to_sell = min(current_position, current_position)  # Sell all
                        self.portfolio.execute_trade(symbol, 'SELL', shares_to_sell, current_price, current_date)
            
            # Calculate daily portfolio value
            if current_prices:
                portfolio_value = self.portfolio.get_portfolio_value(current_prices)
                daily_portfolio_values.append(portfolio_value)
                
                # Calculate daily return
                if i > 0:
                    daily_return = (portfolio_value - daily_portfolio_values[i-1]) / daily_portfolio_values[i-1]
                    self.daily_returns.append(daily_return)
                else:
                    self.daily_returns.append(0.0)
                
                self.equity_curve.append({
                    'date': current_date,
                    'portfolio_value': portfolio_value,
                    'cash': self.portfolio.cash,
                    'positions': len(self.portfolio.positions)
                })
        
        # Calculate final performance
        final_prices = {}
        for symbol, data in filtered_data.items():
            if not data.empty:
                final_prices[symbol] = data['Close'].iloc[-1]
        
        final_metrics = self.portfolio.calculate_performance_metrics(final_prices)
        
        # Add advanced metrics
        advanced_metrics = self.calculate_advanced_metrics()
        
        self.results = {
            'metrics': {**final_metrics, **advanced_metrics},
            'trades': self.portfolio.trades,
            'positions': self.portfolio.positions,
            'equity_curve': self.equity_curve,
            'daily_returns': self.daily_returns,
            'backtest_period': {
                'start_date': start_date,
                'end_date': end_date,
                'total_days': len(all_dates)
            }
        }
        
        print("✅ Backtest completed!")
        return self.results
    
    def calculate_advanced_metrics(self) -> Dict:
        """
        Calculate advanced performance metrics for the backtest.
        """
        if not self.daily_returns:
            return {}
        
        returns = np.array(self.daily_returns)
        
        # Basic statistics
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        
        # Risk metrics
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        sortino_ratio = annualized_return / (np.std(returns[returns < 0]) * np.sqrt(252)) if np.std(returns[returns < 0]) > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate and profit factor
        winning_trades = [t for t in self.portfolio.trades if t.get('status') == 'EXECUTED' and t['action'] == 'SELL']
        if winning_trades:
            # Calculate trade returns
            trade_returns = []
            for trade in winning_trades:
                # Find corresponding buy trade
                buy_trades = [t for t in self.portfolio.trades if t['symbol'] == trade['symbol'] 
                             and t['action'] == 'BUY' and t['status'] == 'EXECUTED']
                if buy_trades:
                    buy_price = buy_trades[-1]['price']
                    trade_return = (trade['price'] - buy_price) / buy_price
                    trade_returns.append(trade_return)
            
            if trade_returns:
                win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns)
                avg_win = np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0
                avg_loss = np.mean([r for r in trade_returns if r < 0]) if any(r < 0 for r in trade_returns) else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            else:
                win_rate = 0
                profit_factor = 0
        else:
            win_rate = 0
            profit_factor = 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Information ratio (assuming risk-free rate of 0.02)
        risk_free_rate = 0.02
        excess_returns = returns - risk_free_rate / 252
        information_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
        
        return {
            'Total_Return': total_return,
            'Annualized_Return': annualized_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Sortino_Ratio': sortino_ratio,
            'Max_Drawdown': max_drawdown,
            'Calmar_Ratio': calmar_ratio,
            'Information_Ratio': information_ratio,
            'Win_Rate': win_rate,
            'Profit_Factor': profit_factor,
            'Total_Trades': len([t for t in self.portfolio.trades if t.get('status') == 'EXECUTED']),
            'Winning_Trades': len([t for t in self.portfolio.trades if t.get('status') == 'EXECUTED' and t['action'] == 'SELL'])
        }
    
    def compare_with_benchmark(self, benchmark_returns: List[float]) -> Dict:
        """
        Compare backtest results with a benchmark (e.g., NIFTY 50).
        """
        if not self.daily_returns or not benchmark_returns:
            return {}
        
        strategy_returns = np.array(self.daily_returns)
        benchmark_returns = np.array(benchmark_returns)
        
        # Ensure same length
        min_length = min(len(strategy_returns), len(benchmark_returns))
        strategy_returns = strategy_returns[:min_length]
        benchmark_returns = benchmark_returns[:min_length]
        
        # Calculate metrics
        strategy_total_return = np.prod(1 + strategy_returns) - 1
        benchmark_total_return = np.prod(1 + benchmark_returns) - 1
        
        strategy_volatility = np.std(strategy_returns) * np.sqrt(252)
        benchmark_volatility = np.std(benchmark_returns) * np.sqrt(252)
        
        strategy_sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
        benchmark_sharpe = np.mean(benchmark_returns) / np.std(benchmark_returns) * np.sqrt(252) if np.std(benchmark_returns) > 0 else 0
        
        # Beta calculation
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Alpha calculation
        risk_free_rate = 0.02
        alpha = (np.mean(strategy_returns) - risk_free_rate / 252) - beta * (np.mean(benchmark_returns) - risk_free_rate / 252)
        alpha_annualized = alpha * 252
        
        return {
            'Strategy_Total_Return': strategy_total_return,
            'Benchmark_Total_Return': benchmark_total_return,
            'Excess_Return': strategy_total_return - benchmark_total_return,
            'Strategy_Volatility': strategy_volatility,
            'Benchmark_Volatility': benchmark_volatility,
            'Strategy_Sharpe': strategy_sharpe,
            'Benchmark_Sharpe': benchmark_sharpe,
            'Beta': beta,
            'Alpha': alpha_annualized,
            'Information_Ratio': (strategy_total_return - benchmark_total_return) / (strategy_volatility - benchmark_volatility) if strategy_volatility != benchmark_volatility else 0
        }
    
    def generate_trade_analysis(self) -> Dict:
        """
        Generate detailed analysis of individual trades.
        """
        executed_trades = [t for t in self.portfolio.trades if t.get('status') == 'EXECUTED']
        
        if not executed_trades:
            return {}
        
        # Group trades by symbol
        trades_by_symbol = {}
        for trade in executed_trades:
            symbol = trade['symbol']
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = []
            trades_by_symbol[symbol].append(trade)
        
        # Calculate trade statistics
        trade_stats = {}
        for symbol, trades in trades_by_symbol.items():
            buy_trades = [t for t in trades if t['action'] == 'BUY']
            sell_trades = [t for t in trades if t['action'] == 'SELL']
            
            total_buy_value = sum(t['value'] for t in buy_trades)
            total_sell_value = sum(t['value'] for t in sell_trades)
            total_commission = sum(t['commission'] for t in trades)
            
            # Calculate P&L for completed trades
            completed_trades = []
            for sell_trade in sell_trades:
                # Find corresponding buy trade
                buy_trades_for_symbol = [t for t in buy_trades if t['symbol'] == sell_trade['symbol']]
                if buy_trades_for_symbol:
                    buy_trade = buy_trades_for_symbol[-1]  # Most recent buy
                    pnl = (sell_trade['price'] - buy_trade['price']) * sell_trade['shares']
                    completed_trades.append(pnl)
            
            trade_stats[symbol] = {
                'total_trades': len(trades),
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'total_buy_value': total_buy_value,
                'total_sell_value': total_sell_value,
                'total_commission': total_commission,
                'total_pnl': sum(completed_trades),
                'avg_pnl_per_trade': np.mean(completed_trades) if completed_trades else 0,
                'winning_trades': len([p for p in completed_trades if p > 0]),
                'losing_trades': len([p for p in completed_trades if p < 0])
            }
        
        return trade_stats
    
    def print_backtest_summary(self):
        """
        Print a comprehensive summary of the backtest results.
        """
        if not self.results:
            print("No backtest results available.")
            return
        
        metrics = self.results['metrics']
        
        print("\n" + "="*60)
        print("📊 BACKTEST SUMMARY")
        print("="*60)
        
        print(f"\n💰 PERFORMANCE METRICS:")
        print(f"   Total Return: {metrics.get('Total_Return', 0):.2%}")
        print(f"   Annualized Return: {metrics.get('Annualized_Return', 0):.2%}")
        print(f"   Volatility: {metrics.get('Volatility', 0):.2%}")
        print(f"   Sharpe Ratio: {metrics.get('Sharpe_Ratio', 0):.2f}")
        print(f"   Sortino Ratio: {metrics.get('Sortino_Ratio', 0):.2f}")
        print(f"   Maximum Drawdown: {metrics.get('Max_Drawdown', 0):.2%}")
        print(f"   Calmar Ratio: {metrics.get('Calmar_Ratio', 0):.2f}")
        
        print(f"\n📈 TRADING METRICS:")
        print(f"   Total Trades: {metrics.get('Total_Trades', 0)}")
        print(f"   Win Rate: {metrics.get('Win_Rate', 0):.2%}")
        print(f"   Profit Factor: {metrics.get('Profit_Factor', 0):.2f}")
        print(f"   Information Ratio: {metrics.get('Information_Ratio', 0):.2f}")
        
        print(f"\n💼 PORTFOLIO STATUS:")
        print(f"   Final Value: ₹{metrics.get('Current_Value', 0):,.2f}")
        print(f"   Cash: ₹{metrics.get('Cash', 0):,.2f}")
        print(f"   Active Positions: {metrics.get('Positions', 0)}")
        
        if self.results.get('backtest_period'):
            period = self.results['backtest_period']
            print(f"\n📅 BACKTEST PERIOD:")
            print(f"   Start Date: {period['start_date'].strftime('%Y-%m-%d')}")
            print(f"   End Date: {period['end_date'].strftime('%Y-%m-%d')}")
            print(f"   Total Days: {period['total_days']}")
        
        print("\n" + "="*60) 