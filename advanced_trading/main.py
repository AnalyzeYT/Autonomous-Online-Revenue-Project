"""
main.py
Main orchestrator and entry points for the Advanced Stock Trading System.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .data import DataCollector
from .model import AdvancedLSTMModel
from .strategy import TradingStrategy
from .portfolio import PortfolioManager
from .risk import RiskManager
from .dashboard import TradingDashboard
from .alerts import AlertSystem
from .backtest import BacktestEngine
from .utils import setup_gpu_tpu, format_currency, format_percentage

class TradingSystem:
    """
    Main trading system orchestrator that coordinates all components.
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.data_collector = DataCollector()
        self.portfolio = PortfolioManager(initial_capital)
        self.risk_manager = RiskManager(initial_capital)
        self.dashboard = TradingDashboard()
        self.alert_system = AlertSystem()
        self.model = None
        self.strategy = None
        self.scaler = None
        self.feature_scaler = None
        self.initial_capital = initial_capital
        
    def initialize_system(self) -> Tuple:
        """
        Initialize the complete trading system.
        Returns strategy and stock list.
        """
        print("🚀 Initializing Advanced Trading System...")
        
        # Setup hardware acceleration
        strategy = setup_gpu_tpu()
        print("✅ Hardware acceleration configured")
        
        # Get stock list
        stocks = self.data_collector.get_indian_stocks_list()
        print(f"📊 Loaded {len(stocks)} Indian stocks for analysis")
        
        return strategy, stocks
    
    def train_model_for_stock(self, symbol: str, strategy) -> Optional[Dict]:
        """
        Train ML model for a specific stock.
        """
        print(f"🤖 Training model for {symbol}...")
        
        # Fetch and process data
        stock_data = self.data_collector.fetch_stock_data(symbol, period='2y')
        if stock_data is None:
            return None
        
        # Calculate technical indicators
        processed_data = self.data_collector.calculate_technical_indicators(stock_data['data'])
        
        # Prepare LSTM data
        X, y, features = self.data_collector.prepare_lstm_data(processed_data)
        
        if len(X) < 100:  # Need sufficient data
            print(f"⚠️ Insufficient data for {symbol}")
            return None
        
        # Split data
        train_size = int(len(X) * 0.8)
        val_size = int(len(X) * 0.1)
        
        X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
        y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]
        
        # Build and train model
        input_shape = (X.shape[1], X.shape[2])
        lstm_model = AdvancedLSTMModel(input_shape, strategy)
        model = lstm_model.build_model()
        
        # Train model
        history = lstm_model.train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
        
        # Evaluate model
        metrics, predictions, actual = lstm_model.evaluate_model(X_test, y_test, self.data_collector.scaler)
        
        print(f"✅ Model trained for {symbol}")
        print(f"   RMSE: {metrics['RMSE']:.2f}")
        print(f"   Directional Accuracy: {metrics['Directional_Accuracy']:.1f}%")
        
        return {
            'model': lstm_model,
            'history': history,
            'metrics': metrics,
            'predictions': predictions,
            'actual': actual,
            'features': features
        }
    
    def run_live_trading_simulation(self, symbol: str, model_results: Dict) -> Optional[Dict]:
        """
        Run live trading simulation for a stock.
        """
        print(f"📈 Starting live trading simulation for {symbol}...")
        
        # Initialize strategy
        self.strategy = TradingStrategy(
            model_results['model'],
            self.data_collector.scaler,
            self.data_collector.feature_scaler
        )
        
        # Fetch recent data
        recent_data = self.data_collector.fetch_stock_data(symbol, period='3mo', interval='1d')
        if recent_data is None:
            return
        
        processed_data = self.data_collector.calculate_technical_indicators(recent_data['data'])
        
        # Generate signals
        signal_data = self.strategy.generate_signals(processed_data, symbol)
        
        if signal_data:
            print(f"🎯 Signal Generated for {symbol}:")
            print(f"   Signal: {signal_data['signal']}")
            print(f"   Confidence: {signal_data['confidence']:.2%}")
            print(f"   Current Price: ₹{signal_data['current_price']:.2f}")
            print(f"   Predicted Price: ₹{signal_data['predicted_price']:.2f}")
            print(f"   Expected Change: {signal_data['price_change_pct']:.2%}")
            
            # Check for alerts
            alerts = self.alert_system.check_alerts(signal_data, processed_data)
            if alerts:
                print("🚨 Alerts triggered:")
                for alert in alerts:
                    print(f"   {alert['type']}: {alert['message']}")
            
            # Execute trade if conditions are met
            if signal_data['signal'] in ['BUY', 'SELL'] and signal_data['confidence'] > 0.7:
                current_price = signal_data['current_price']
                
                if signal_data['signal'] == 'BUY':
                    shares = self.risk_manager.calculate_position_size(current_price)
                    if shares > 0:
                        risk_ok, risk_msg = self.risk_manager.check_risk_limits(symbol, current_price, shares)
                        if risk_ok:
                            trade = self.portfolio.execute_trade(symbol, 'BUY', shares, current_price, datetime.now())
                            print(f"✅ Executed BUY order: {trade}")
                        else:
                            print(f"❌ Trade rejected: {risk_msg}")
                
                elif signal_data['signal'] == 'SELL':
                    current_position = self.portfolio.positions.get(symbol, 0)
                    if current_position > 0:
                        trade = self.portfolio.execute_trade(symbol, 'SELL', current_position, current_price, datetime.now())
                        print(f"✅ Executed SELL order: {trade}")
        
        return signal_data
    
    def generate_comprehensive_report(self, symbol: str, model_results: Dict, 
                                    signal_data: Optional[Dict] = None) -> Dict:
        """
        Generate comprehensive analysis report for a stock.
        """
        print(f"📊 Generating comprehensive report for {symbol}...")
        
        # Fetch data for visualization
        stock_data = self.data_collector.fetch_stock_data(symbol, period='6mo')
        if stock_data is None:
            return {}
        
        processed_data = self.data_collector.calculate_technical_indicators(stock_data['data'])
        
        # Create visualizations
        print("📈 Creating price prediction chart...")
        price_fig = self.dashboard.plot_price_prediction(
            model_results['actual'],
            model_results['predictions'],
            symbol
        )
        
        print("📊 Creating technical analysis chart...")
        tech_fig = self.dashboard.plot_technical_indicators(processed_data.tail(100), symbol)
        
        print("📈 Creating training history chart...")
        training_fig = self.dashboard.plot_model_training_history(model_results['history'])
        
        # Portfolio performance
        current_prices = {symbol: processed_data['Close'].iloc[-1]}
        portfolio_metrics = self.portfolio.calculate_performance_metrics(current_prices)
        
        print("💼 Creating portfolio performance chart...")
        portfolio_fig = self.dashboard.plot_portfolio_performance({
            'trades': self.portfolio.trades,
            'positions': self.portfolio.positions
        })
        
        # Display results
        print("\n" + "="*60)
        print(f"📊 COMPREHENSIVE ANALYSIS REPORT - {symbol}")
        print("="*60)
        
        print(f"\n🤖 MODEL PERFORMANCE:")
        print(f"   RMSE: ₹{model_results['metrics']['RMSE']:.2f}")
        print(f"   MAE: ₹{model_results['metrics']['MAE']:.2f}")
        print(f"   Directional Accuracy: {model_results['metrics']['Directional_Accuracy']:.1f}%")
        
        if signal_data:
            print(f"\n🎯 CURRENT SIGNAL:")
            print(f"   Signal: {signal_data['signal']}")
            print(f"   Confidence: {signal_data['confidence']:.2%}")
            print(f"   Current Price: ₹{signal_data['current_price']:.2f}")
            print(f"   Predicted Price: ₹{signal_data['predicted_price']:.2f}")
            print(f"   Expected Change: {signal_data['price_change_pct']:.2%}")
            
            print(f"\n📊 TECHNICAL INDICATORS:")
            ti = signal_data['technical_indicators']
            print(f"   RSI: {ti['RSI']:.1f}")
            print(f"   MACD: {ti['MACD']:.4f}")
            print(f"   MACD Signal: {ti['MACD_Signal']:.4f}")
            print(f"   SMA 20: ₹{ti['SMA_20']:.2f}")
            print(f"   SMA 50: ₹{ti['SMA_50']:.2f}")
            print(f"   BB Position: {ti['BB_Position']:.2f}")
        
        print(f"\n💼 PORTFOLIO STATUS:")
        print(f"   Total Value: {format_currency(portfolio_metrics['Current_Value'])}")
        print(f"   Total Return: {format_percentage(portfolio_metrics['Total_Return'])}")
        print(f"   Cash: {format_currency(portfolio_metrics['Cash'])}")
        print(f"   Active Positions: {portfolio_metrics['Positions']}")
        print(f"   Total Trades: {portfolio_metrics['Total_Trades']}")
        print(f"   Sharpe Ratio: {portfolio_metrics['Sharpe_Ratio']:.2f}")
        print(f"   Max Drawdown: {format_percentage(portfolio_metrics['Max_Drawdown'])}")
        
        # Recent alerts
        recent_alerts = self.alert_system.get_recent_alerts(24)
        if recent_alerts:
            print(f"\n🚨 RECENT ALERTS (24h):")
            for alert in recent_alerts[-5:]:  # Show last 5 alerts
                print(f"   {alert['priority']} - {alert['type']}: {alert['message']}")
        
        print("\n" + "="*60)
        
        return {
            'model_metrics': model_results['metrics'],
            'signal_data': signal_data,
            'portfolio_metrics': portfolio_metrics,
            'alerts': recent_alerts,
            'charts': {
                'price_prediction': price_fig,
                'technical_analysis': tech_fig,
                'training_history': training_fig,
                'portfolio_performance': portfolio_fig
            }
        }

class BatchProcessor:
    """
    Process multiple stocks in parallel for efficient analysis.
    """
    
    def __init__(self, trading_system: TradingSystem):
        self.trading_system = trading_system
        self.results = {}
    
    def process_stock_batch(self, symbols: List[str], max_workers: int = 4) -> Dict:
        """
        Process multiple stocks in parallel.
        """
        print(f"🔄 Processing {len(symbols)} stocks in parallel...")
        
        strategy, _ = self.trading_system.initialize_system()
        
        def process_single_stock(symbol: str) -> Tuple[str, Optional[Dict]]:
            try:
                print(f"Processing {symbol}...")
                
                # Train model
                model_results = self.trading_system.train_model_for_stock(symbol, strategy)
                if model_results is None:
                    return symbol, None
                
                # Run trading simulation
                signal_data = self.trading_system.run_live_trading_simulation(symbol, model_results)
                
                # Generate report
                report = self.trading_system.generate_comprehensive_report(symbol, model_results, signal_data)
                
                return symbol, {
                    'model_results': model_results,
                    'signal_data': signal_data,
                    'report': report
                }
                
            except Exception as e:
                print(f"❌ Error processing {symbol}: {e}")
                return symbol, None
        
        # Process stocks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_stock, symbol) for symbol in symbols]
            
            for future in as_completed(futures):
                symbol, result = future.result()
                self.results[symbol] = result
                
                if result:
                    print(f"✅ Completed processing {symbol}")
                else:
                    print(f"❌ Failed to process {symbol}")
        
        return self.results
    
    def generate_batch_summary(self) -> Dict:
        """
        Generate summary of batch processing results.
        """
        successful = [s for s, r in self.results.items() if r is not None]
        failed = [s for s, r in self.results.items() if r is None]
        
        print("\n" + "="*60)
        print("📊 BATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"✅ Successfully processed: {len(successful)} stocks")
        print(f"❌ Failed to process: {len(failed)} stocks")
        
        if successful:
            print(f"\n🎯 TOP SIGNALS:")
            signals = []
            for symbol in successful:
                if self.results[symbol]['signal_data']:
                    signals.append((symbol, self.results[symbol]['signal_data']))
            
            # Sort by confidence
            signals.sort(key=lambda x: x[1]['confidence'], reverse=True)
            
            for symbol, signal in signals[:5]:  # Top 5 signals
                print(f"   {symbol}: {signal['signal']} (Confidence: {signal['confidence']:.2%})")
        
        return {
            'successful': successful,
            'failed': failed,
            'results': self.results
        }

def main():
    """
    Main execution function for the trading system.
    """
    print("🚀 ADVANCED STOCK TRADING SYSTEM FOR INDIAN MARKETS")
    print("=" * 60)
    
    # Initialize trading system
    trading_system = TradingSystem(initial_capital=100000)
    
    # Get stock symbols (start with a few for demonstration)
    demo_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
    
    # Initialize batch processor
    batch_processor = BatchProcessor(trading_system)
    
    # Process stocks
    print(f"🔄 Starting batch processing of {len(demo_stocks)} stocks...")
    results = batch_processor.process_stock_batch(demo_stocks, max_workers=2)
    
    # Generate summary
    summary = batch_processor.generate_batch_summary()
    
    print("\n🎉 SYSTEM EXECUTION COMPLETED!")
    print("📊 Check the generated charts and reports above for detailed analysis.")
    
    return trading_system, batch_processor, results, summary

def quick_analysis(symbol: str, period: str = '6mo') -> Tuple[Optional[pd.DataFrame], Optional]:
    """
    Quick analysis function for individual stocks.
    """
    print(f"🔍 Quick Analysis for {symbol}")
    print("-" * 40)
    
    # Initialize components
    data_collector = DataCollector()
    dashboard = TradingDashboard()
    
    # Fetch and process data
    stock_data = data_collector.fetch_stock_data(symbol, period=period)
    if stock_data is None:
        print(f"❌ Could not fetch data for {symbol}")
        return None, None
    
    processed_data = data_collector.calculate_technical_indicators(stock_data['data'])
    
    # Current metrics
    current = processed_data.iloc[-1]
    print(f"💰 Current Price: ₹{current['Close']:.2f}")
    print(f"📊 RSI: {current['RSI']:.1f}")
    print(f"📈 MACD: {current['MACD']:.4f}")
    print(f"📉 MACD Signal: {current['MACD_signal']:.4f}")
    print(f"🔄 Volume: {current['Volume']:,.0f}")
    
    # Generate technical chart
    fig = dashboard.plot_technical_indicators(processed_data.tail(100), symbol)
    
    try:
        fig.show()
    except:
        print("📊 Chart generated (display requires interactive environment)")
    
    return processed_data, fig

def run_backtest_demo() -> Optional[Dict]:
    """
    Run a demonstration backtest.
    """
    print("🔄 Running Backtest Demo...")
    print("-" * 40)
    
    # Initialize components
    trading_system = TradingSystem()
    data_collector = DataCollector()
    
    # Fetch data for backtesting
    symbols = ['RELIANCE.NS', 'TCS.NS']
    data_dict = {}
    
    for symbol in symbols:
        stock_data = data_collector.fetch_stock_data(symbol, period='1y')
        if stock_data:
            processed_data = data_collector.calculate_technical_indicators(stock_data['data'])
            data_dict[symbol] = processed_data
    
    if not data_dict:
        print("❌ No data available for backtesting")
        return None
    
    # Create mock strategy for demo
    class MockStrategy:
        def generate_signals(self, data, symbol):
            if len(data) < 20:
                return None
            
            current = data.iloc[-1]
            signal = 'BUY' if current['RSI'] < 40 else 'SELL' if current['RSI'] > 60 else 'HOLD'
            
            return {
                'signal': signal,
                'confidence': 0.8 if signal != 'HOLD' else 0.3,
                'current_price': current['Close'],
                'predicted_price': current['Close'] * 1.02,
                'price_change_pct': 0.02,
                'technical_indicators': {
                    'RSI': current['RSI'],
                    'MACD': current['MACD'],
                    'MACD_Signal': current['MACD_signal'],
                    'SMA_20': current['SMA_20'],
                    'SMA_50': current['SMA_50'],
                    'BB_Position': 0.5
                }
            }
    
    # Run backtest
    backtest_engine = BacktestEngine(initial_capital=100000)
    strategy = MockStrategy()
    
    start_date = pd.Timestamp.now() - pd.Timedelta(days=180)
    end_date = pd.Timestamp.now() - pd.Timedelta(days=30)
    
    results = backtest_engine.run_backtest(strategy, data_dict, start_date, end_date)
    
    print("📊 Backtest Results:")
    print(f"   Final Portfolio Value: {format_currency(results['metrics']['Current_Value'])}")
    print(f"   Total Return: {format_percentage(results['metrics']['Total_Return'])}")
    print(f"   Total Trades: {results['metrics']['Total_Trades']}")
    print(f"   Sharpe Ratio: {results['metrics']['Sharpe_Ratio']:.2f}")
    
    return results

if __name__ == "__main__":
    # Uncomment the desired function to run
    
    # Run full system
    # main()
    
    # Quick analysis of a single stock
    # quick_analysis('RELIANCE.NS')
    
    # Run backtest demo
    # run_backtest_demo()
    
    print("🎯 Advanced Stock Trading System is ready!")
    print("📚 Available functions:")
    print("   - main(): Run complete system")
    print("   - quick_analysis('SYMBOL'): Quick stock analysis")
    print("   - run_backtest_demo(): Run backtesting demo")
    print("\n🚀 Uncomment the desired function in the main block to execute!") 