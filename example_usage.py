"""
example_usage.py
Example usage of the Advanced Stock Trading System.
"""

# Example 1: Basic Usage
def basic_example():
    """Basic usage example for quick analysis."""
    from advanced_trading import quick_analysis
    
    # Quick analysis of a stock
    data, chart = quick_analysis('RELIANCE.NS')
    print("Basic analysis completed!")

# Example 2: Full System Usage
def full_system_example():
    """Full system usage with model training and trading simulation."""
    from advanced_trading import TradingSystem, BatchProcessor
    
    # Initialize the trading system
    trading_system = TradingSystem(initial_capital=100000)
    
    # Get stock list
    stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
    
    # Process stocks in batch
    batch_processor = BatchProcessor(trading_system)
    results = batch_processor.process_stock_batch(stocks, max_workers=2)
    
    # Generate summary
    summary = batch_processor.generate_batch_summary()
    print("Full system analysis completed!")

# Example 3: Individual Component Usage
def component_example():
    """Example of using individual components."""
    from advanced_trading import DataCollector, TradingDashboard, AlertSystem
    
    # Data collection
    dc = DataCollector()
    stock_data = dc.fetch_stock_data('RELIANCE.NS', period='1y')
    if stock_data:
        processed_data = dc.calculate_technical_indicators(stock_data['data'])
        
        # Create dashboard
        dashboard = TradingDashboard()
        fig = dashboard.plot_technical_indicators(processed_data.tail(100), 'RELIANCE.NS')
        
        # Alert system
        alert_system = AlertSystem()
        print("Component usage completed!")

# Example 4: Backtesting
def backtesting_example():
    """Example of running backtests."""
    from advanced_trading import run_backtest_demo
    
    # Run backtest demo
    results = run_backtest_demo()
    print("Backtesting completed!")

# Example 5: Custom Strategy
def custom_strategy_example():
    """Example of creating a custom trading strategy."""
    from advanced_trading import DataCollector, AdvancedLSTMModel, TradingStrategy
    from advanced_trading.utils import setup_gpu_tpu
    
    # Setup
    strategy = setup_gpu_tpu()
    dc = DataCollector()
    
    # Get data
    stock_data = dc.fetch_stock_data('RELIANCE.NS', period='2y')
    if stock_data:
        processed_data = dc.calculate_technical_indicators(stock_data['data'])
        X, y, features = dc.prepare_lstm_data(processed_data)
        
        # Train model
        if len(X) > 100:
            train_size = int(len(X) * 0.8)
            X_train, y_train = X[:train_size], y[:train_size]
            
            lstm_model = AdvancedLSTMModel((X.shape[1], X.shape[2]), strategy)
            model = lstm_model.build_model()
            history = lstm_model.train_model(X_train, y_train, X_train, y_train, epochs=10)
            
            # Create strategy
            trading_strategy = TradingStrategy(lstm_model, dc.scaler, dc.feature_scaler)
            signal = trading_strategy.generate_signals(processed_data, 'RELIANCE.NS')
            
            print("Custom strategy completed!")

# Example 6: Portfolio Management
def portfolio_example():
    """Example of portfolio management."""
    from advanced_trading import PortfolioManager, RiskManager
    
    # Initialize portfolio and risk management
    portfolio = PortfolioManager(initial_capital=100000)
    risk_manager = RiskManager(initial_capital=100000)
    
    # Simulate some trades
    portfolio.execute_trade('RELIANCE.NS', 'BUY', 100, 2500.0, datetime.now())
    portfolio.execute_trade('TCS.NS', 'BUY', 50, 3500.0, datetime.now())
    
    # Calculate performance
    current_prices = {'RELIANCE.NS': 2600.0, 'TCS.NS': 3600.0}
    metrics = portfolio.calculate_performance_metrics(current_prices)
    
    print(f"Portfolio Value: ₹{metrics['Current_Value']:,.2f}")
    print(f"Total Return: {metrics['Total_Return']:.2%}")

# Example 7: Alert System
def alert_example():
    """Example of using the alert system."""
    from advanced_trading import AlertSystem
    
    alert_system = AlertSystem()
    
    # Simulate a signal
    signal_data = {
        'symbol': 'RELIANCE.NS',
        'price_change_pct': 0.08,  # 8% change
        'confidence': 0.85,
        'technical_indicators': {
            'RSI': 20,  # Oversold
            'MACD': 0.5,
            'MACD_Signal': 0.3,
            'SMA_20': 2500,
            'SMA_50': 2450,
            'BB_Position': 0.1,
            'Volume_Ratio': 2.5
        }
    }
    
    # Check for alerts
    alerts = alert_system.check_alerts(signal_data, None)
    
    for alert in alerts:
        print(f"Alert: {alert['type']} - {alert['message']}")

# Example 8: Dashboard Visualization
def dashboard_example():
    """Example of creating visualizations."""
    from advanced_trading import DataCollector, TradingDashboard
    
    dc = DataCollector()
    dashboard = TradingDashboard()
    
    # Get data
    stock_data = dc.fetch_stock_data('RELIANCE.NS', period='6mo')
    if stock_data:
        processed_data = dc.calculate_technical_indicators(stock_data['data'])
        
        # Create various charts
        tech_chart = dashboard.plot_technical_indicators(processed_data, 'RELIANCE.NS')
        
        # Simulate portfolio data
        portfolio_data = {
            'trades': [],
            'positions': {}
        }
        portfolio_chart = dashboard.plot_portfolio_performance(portfolio_data)
        
        print("Dashboard visualizations created!")

# Example 9: Risk Management
def risk_example():
    """Example of risk management features."""
    from advanced_trading import RiskManager
    import numpy as np
    
    risk_manager = RiskManager(initial_capital=100000)
    
    # Calculate position size
    shares = risk_manager.calculate_position_size(2500.0, volatility=0.02)
    print(f"Recommended shares: {shares}")
    
    # Check risk limits
    risk_ok, message = risk_manager.check_risk_limits('RELIANCE.NS', 2500.0, shares)
    print(f"Risk check: {message}")
    
    # Calculate VaR
    returns = np.random.normal(0.001, 0.02, 100)  # Simulated returns
    var = risk_manager.calculate_var(returns, confidence_level=0.95)
    print(f"Value at Risk (95%): {var:.4f}")

# Example 10: Complete Workflow
def complete_workflow_example():
    """Complete workflow example combining all components."""
    from advanced_trading import TradingSystem, BatchProcessor
    
    print("🚀 Starting Complete Workflow Example")
    
    # Initialize system
    trading_system = TradingSystem(initial_capital=100000)
    strategy, stocks = trading_system.initialize_system()
    
    # Select a few stocks for analysis
    selected_stocks = stocks[:3]  # First 3 stocks
    
    # Process stocks
    batch_processor = BatchProcessor(trading_system)
    results = batch_processor.process_stock_batch(selected_stocks, max_workers=2)
    
    # Generate summary
    summary = batch_processor.generate_batch_summary()
    
    print("✅ Complete workflow finished!")

if __name__ == "__main__":
    print("🎯 Advanced Stock Trading System - Example Usage")
    print("=" * 50)
    
    # Run examples
    print("\n1. Basic Example:")
    # basic_example()
    
    print("\n2. Component Example:")
    # component_example()
    
    print("\n3. Portfolio Example:")
    # portfolio_example()
    
    print("\n4. Alert Example:")
    # alert_example()
    
    print("\n5. Dashboard Example:")
    # dashboard_example()
    
    print("\n6. Risk Management Example:")
    # risk_example()
    
    print("\n7. Complete Workflow:")
    # complete_workflow_example()
    
    print("\n📚 Uncomment the examples above to run them!")
    print("🚀 Each example demonstrates different aspects of the system.") 