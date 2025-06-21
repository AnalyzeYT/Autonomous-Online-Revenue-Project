"""
streamlit_app.py
Streamlit web application for the Advanced Stock Trading System.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional

# Import our trading system components
from .data import DataCollector
from .model import AdvancedLSTMModel
from .strategy import TradingStrategy
from .portfolio import PortfolioManager
from .risk import RiskManager
from .dashboard import TradingDashboard
from .alerts import AlertSystem
from .backtest import BacktestEngine
from .sentiment import SentimentAnalyzer
from .optimization import PortfolioOptimizer
from .utils import setup_gpu_tpu, format_currency, format_percentage

class StreamlitTradingApp:
    """
    Streamlit web application for the trading system.
    """
    
    def __init__(self):
        self.data_collector = DataCollector()
        self.dashboard = TradingDashboard()
        self.alert_system = AlertSystem()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.portfolio_optimizer = PortfolioOptimizer()
        
        # Initialize session state
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = PortfolioManager(initial_capital=100000)
        if 'risk_manager' not in st.session_state:
            st.session_state.risk_manager = RiskManager(initial_capital=100000)
        if 'trading_signals' not in st.session_state:
            st.session_state.trading_signals = []
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
    
    def run(self):
        """
        Run the Streamlit application.
        """
        st.set_page_config(
            page_title="Advanced Stock Trading System",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .alert-high {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
        }
        .alert-medium {
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
        }
        .alert-low {
            background-color: #e8f5e8;
            border-left: 4px solid #4caf50;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown('<h1 class="main-header">📈 Advanced Stock Trading System</h1>', unsafe_allow_html=True)
        
        # Sidebar
        self.sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Market Overview", "🤖 ML Analysis", "💼 Portfolio", 
            "📈 Technical Analysis", "⚙️ Optimization", "🚨 Alerts"
        ])
        
        with tab1:
            self.market_overview_tab()
        
        with tab2:
            self.ml_analysis_tab()
        
        with tab3:
            self.portfolio_tab()
        
        with tab4:
            self.technical_analysis_tab()
        
        with tab5:
            self.optimization_tab()
        
        with tab6:
            self.alerts_tab()
    
    def sidebar(self):
        """
        Create the sidebar with controls.
        """
        st.sidebar.title("🎛️ Trading Controls")
        
        # Stock selection
        st.sidebar.subheader("📈 Stock Selection")
        stocks = self.data_collector.get_indian_stocks_list()
        selected_stock = st.sidebar.selectbox(
            "Select Stock",
            stocks,
            index=0
        )
        
        # Time period
        st.sidebar.subheader("⏰ Time Period")
        period = st.sidebar.selectbox(
            "Select Period",
            ["1mo", "3mo", "6mo", "1y", "2y"],
            index=2
        )
        
        # Analysis parameters
        st.sidebar.subheader("🔧 Analysis Parameters")
        confidence_threshold = st.sidebar.slider(
            "Signal Confidence Threshold",
            min_value=0.5,
            max_value=0.95,
            value=0.7,
            step=0.05
        )
        
        risk_tolerance = st.sidebar.slider(
            "Risk Tolerance",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1
        )
        
        # Portfolio settings
        st.sidebar.subheader("💼 Portfolio Settings")
        initial_capital = st.sidebar.number_input(
            "Initial Capital (₹)",
            min_value=10000,
            max_value=1000000,
            value=100000,
            step=10000
        )
        
        max_position_size = st.sidebar.slider(
            "Max Position Size (%)",
            min_value=5,
            max_value=50,
            value=10,
            step=5
        )
        
        # Action buttons
        st.sidebar.subheader("🚀 Actions")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🔄 Refresh Data"):
                st.session_state.refresh_data = True
                st.rerun()
        
        with col2:
            if st.button("📊 Run Analysis"):
                st.session_state.run_analysis = True
                st.rerun()
        
        # Store selected parameters
        st.session_state.selected_stock = selected_stock
        st.session_state.period = period
        st.session_state.confidence_threshold = confidence_threshold
        st.session_state.risk_tolerance = risk_tolerance
        st.session_state.initial_capital = initial_capital
        st.session_state.max_position_size = max_position_size
    
    def market_overview_tab(self):
        """
        Market overview tab with real-time data.
        """
        st.header("📊 Market Overview")
        
        # Get selected stock
        selected_stock = st.session_state.get('selected_stock', 'RELIANCE.NS')
        
        # Fetch data
        with st.spinner("Fetching market data..."):
            stock_data = self.data_collector.fetch_stock_data(selected_stock, period=st.session_state.get('period', '6mo'))
        
        if stock_data is None:
            st.error("Failed to fetch data for the selected stock.")
            return
        
        # Process data
        processed_data = self.data_collector.calculate_technical_indicators(stock_data['data'])
        
        # Display current metrics
        col1, col2, col3, col4 = st.columns(4)
        
        current = processed_data.iloc[-1]
        
        with col1:
            st.metric(
                "Current Price",
                f"₹{current['Close']:.2f}",
                f"{current['Price_Change']*100:.2f}%"
            )
        
        with col2:
            st.metric(
                "RSI",
                f"{current['RSI']:.1f}",
                "Oversold" if current['RSI'] < 30 else "Overbought" if current['RSI'] > 70 else "Neutral"
            )
        
        with col3:
            st.metric(
                "MACD",
                f"{current['MACD']:.4f}",
                "Bullish" if current['MACD'] > current['MACD_signal'] else "Bearish"
            )
        
        with col4:
            st.metric(
                "Volume",
                f"{current['Volume']:,.0f}",
                f"{current['Volume_Change']*100:.1f}%"
            )
        
        # Price chart
        st.subheader("📈 Price Chart")
        
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=processed_data.index,
            open=processed_data['Open'],
            high=processed_data['High'],
            low=processed_data['Low'],
            close=processed_data['Close'],
            name='Price'
        ))
        
        # Add moving averages
        if 'SMA_20' in processed_data.columns:
            fig.add_trace(go.Scatter(
                x=processed_data.index,
                y=processed_data['SMA_20'],
                name='SMA 20',
                line=dict(color='orange', width=1)
            ))
        
        if 'SMA_50' in processed_data.columns:
            fig.add_trace(go.Scatter(
                x=processed_data.index,
                y=processed_data['SMA_50'],
                name='SMA 50',
                line=dict(color='blue', width=1)
            ))
        
        fig.update_layout(
            title=f"{selected_stock} - Price Chart",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Market sentiment
        st.subheader("😊 Market Sentiment")
        
        with st.spinner("Analyzing sentiment..."):
            sentiment = self.sentiment_analyzer.analyze_stock_sentiment(selected_stock, days=3)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment_color = {
                'positive': 'green',
                'negative': 'red',
                'neutral': 'gray'
            }.get(sentiment['overall_sentiment'], 'gray')
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>News Sentiment</h3>
                <p style="color: {sentiment_color}; font-size: 1.5rem; font-weight: bold;">
                    {sentiment['overall_sentiment'].title()}
                </p>
                <p>Confidence: {sentiment['confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            social_sentiment = self.sentiment_analyzer.analyze_social_media_sentiment(selected_stock)
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Social Sentiment</h3>
                <p style="color: {sentiment_color}; font-size: 1.5rem; font-weight: bold;">
                    {social_sentiment['social_sentiment'].title()}
                </p>
                <p>Posts: {social_sentiment['posts_analyzed']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Fear & Greed Index (simulated)
            fear_greed = self.sentiment_analyzer.calculate_fear_greed_index({
                'volatility': current['ATR'] / current['Close'],
                'momentum': current['Price_Change'],
                'volume_ratio': current['Volume'] / processed_data['Volume'].rolling(20).mean().iloc[-1],
                'put_call_ratio': 1.0,
                'junk_bond_demand': 0.5,
                'market_momentum': current['Price_Change']
            })
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Fear & Greed</h3>
                <p style="font-size: 1.5rem; font-weight: bold;">
                    {fear_greed['fear_greed_index']:.0f}
                </p>
                <p>{fear_greed['sentiment']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def ml_analysis_tab(self):
        """
        Machine learning analysis tab.
        """
        st.header("🤖 Machine Learning Analysis")
        
        selected_stock = st.session_state.get('selected_stock', 'RELIANCE.NS')
        
        # Model training section
        st.subheader("🧠 Model Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Model Type",
                ["LSTM", "GRU", "Transformer"],
                index=0
            )
            
            epochs = st.slider(
                "Training Epochs",
                min_value=10,
                max_value=200,
                value=50,
                step=10
            )
        
        with col2:
            lookback_period = st.slider(
                "Lookback Period",
                min_value=30,
                max_value=120,
                value=60,
                step=10
            )
            
            forecast_horizon = st.slider(
                "Forecast Horizon",
                min_value=1,
                max_value=30,
                value=1,
                step=1
            )
        
        # Train model button
        if st.button("🚀 Train Model"):
            with st.spinner("Training model..."):
                # Setup GPU/TPU
                strategy = setup_gpu_tpu()
                
                # Initialize trading system
                trading_system = TradingSystem(initial_capital=100000)
                
                # Train model
                model_results = trading_system.train_model_for_stock(selected_stock, strategy)
                
                if model_results:
                    st.session_state.model_results = model_results
                    st.success("Model trained successfully!")
                else:
                    st.error("Failed to train model.")
        
        # Display model results
        if 'model_results' in st.session_state:
            model_results = st.session_state.model_results
            
            st.subheader("📊 Model Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "RMSE",
                    f"₹{model_results['metrics']['RMSE']:.2f}"
                )
            
            with col2:
                st.metric(
                    "MAE",
                    f"₹{model_results['metrics']['MAE']:.2f}"
                )
            
            with col3:
                st.metric(
                    "Directional Accuracy",
                    f"{model_results['metrics']['Directional_Accuracy']:.1f}%"
                )
            
            with col4:
                st.metric(
                    "MSE",
                    f"{model_results['metrics']['MSE']:.4f}"
                )
            
            # Training history chart
            st.subheader("📈 Training History")
            
            if hasattr(model_results['history'], 'history'):
                history = model_results['history'].history
                
                fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss', 'Metrics'))
                
                epochs_range = range(1, len(history['loss']) + 1)
                
                fig.add_trace(go.Scatter(
                    x=list(epochs_range),
                    y=history['loss'],
                    name='Training Loss',
                    line=dict(color='red')
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=list(epochs_range),
                    y=history['val_loss'],
                    name='Validation Loss',
                    line=dict(color='blue')
                ), row=1, col=1)
                
                if 'mae' in history:
                    fig.add_trace(go.Scatter(
                        x=list(epochs_range),
                        y=history['mae'],
                        name='Training MAE',
                        line=dict(color='green')
                    ), row=1, col=2)
                    
                    fig.add_trace(go.Scatter(
                        x=list(epochs_range),
                        y=history['val_mae'],
                        name='Validation MAE',
                        line=dict(color='orange')
                    ), row=1, col=2)
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Prediction vs Actual
            st.subheader("🎯 Predictions vs Actual")
            
            if 'predictions' in model_results and 'actual' in model_results:
                fig = go.Figure()
                
                dates = pd.date_range(start='2023-01-01', periods=len(model_results['actual']), freq='D')
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=model_results['actual'].flatten(),
                    name='Actual',
                    line=dict(color='blue')
                ))
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=model_results['predictions'].flatten(),
                    name='Predicted',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title="Price Predictions vs Actual",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def portfolio_tab(self):
        """
        Portfolio management tab.
        """
        st.header("💼 Portfolio Management")
        
        portfolio = st.session_state.portfolio
        
        # Portfolio overview
        st.subheader("📊 Portfolio Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Simulate current prices for portfolio calculation
        current_prices = {}
        for symbol in portfolio.positions.keys():
            stock_data = self.data_collector.fetch_stock_data(symbol, period='1d')
            if stock_data:
                current_prices[symbol] = stock_data['data']['Close'].iloc[-1]
        
        portfolio_metrics = portfolio.calculate_performance_metrics(current_prices)
        
        with col1:
            st.metric(
                "Total Value",
                format_currency(portfolio_metrics['Current_Value']),
                f"{portfolio_metrics['Total_Return']:.2%}"
            )
        
        with col2:
            st.metric(
                "Cash",
                format_currency(portfolio_metrics['Cash'])
            )
        
        with col3:
            st.metric(
                "Positions",
                portfolio_metrics['Positions']
            )
        
        with col4:
            st.metric(
                "Sharpe Ratio",
                f"{portfolio_metrics['Sharpe_Ratio']:.2f}"
            )
        
        # Portfolio composition
        st.subheader("📈 Portfolio Composition")
        
        if portfolio.positions:
            positions_data = []
            for symbol, shares in portfolio.positions.items():
                price = current_prices.get(symbol, 0)
                value = shares * price
                positions_data.append({
                    'Symbol': symbol,
                    'Shares': shares,
                    'Price': price,
                    'Value': value,
                    'Weight': value / portfolio_metrics['Current_Value'] if portfolio_metrics['Current_Value'] > 0 else 0
                })
            
            positions_df = pd.DataFrame(positions_data)
            st.dataframe(positions_df, use_container_width=True)
            
            # Portfolio pie chart
            fig = go.Figure(data=[go.Pie(
                labels=positions_df['Symbol'],
                values=positions_df['Value'],
                hole=0.3
            )])
            
            fig.update_layout(title="Portfolio Allocation")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No positions in portfolio.")
        
        # Trading history
        st.subheader("📋 Trading History")
        
        if portfolio.trades:
            trades_df = pd.DataFrame(portfolio.trades)
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.info("No trading history available.")
        
        # Risk metrics
        st.subheader("⚠️ Risk Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Max Drawdown",
                format_percentage(portfolio_metrics['Max_Drawdown'])
            )
        
        with col2:
            st.metric(
                "Total Trades",
                portfolio_metrics['Total_Trades']
            )
        
        with col3:
            # Calculate beta (simplified)
            beta = 1.0  # Placeholder
            st.metric(
                "Beta",
                f"{beta:.2f}"
            )
    
    def technical_analysis_tab(self):
        """
        Technical analysis tab.
        """
        st.header("📈 Technical Analysis")
        
        selected_stock = st.session_state.get('selected_stock', 'RELIANCE.NS')
        
        # Fetch data
        stock_data = self.data_collector.fetch_stock_data(selected_stock, period=st.session_state.get('period', '6mo'))
        
        if stock_data is None:
            st.error("Failed to fetch data.")
            return
        
        processed_data = self.data_collector.calculate_technical_indicators(stock_data['data'])
        
        # Technical indicators overview
        st.subheader("📊 Technical Indicators")
        
        current = processed_data.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # RSI
            rsi_color = "red" if current['RSI'] > 70 else "green" if current['RSI'] < 30 else "orange"
            st.markdown(f"""
            <div class="metric-card">
                <h3>RSI</h3>
                <p style="color: {rsi_color}; font-size: 1.5rem; font-weight: bold;">
                    {current['RSI']:.1f}
                </p>
                <p>{'Overbought' if current['RSI'] > 70 else 'Oversold' if current['RSI'] < 30 else 'Neutral'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # MACD
            macd_signal = "Bullish" if current['MACD'] > current['MACD_signal'] else "Bearish"
            macd_color = "green" if current['MACD'] > current['MACD_signal'] else "red"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>MACD</h3>
                <p style="color: {macd_color}; font-size: 1.5rem; font-weight: bold;">
                    {current['MACD']:.4f}
                </p>
                <p>{macd_signal}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Bollinger Bands
            bb_position = (current['Close'] - current['BB_lower']) / (current['BB_upper'] - current['BB_lower'])
            bb_signal = "Overbought" if bb_position > 0.8 else "Oversold" if bb_position < 0.2 else "Neutral"
            bb_color = "red" if bb_position > 0.8 else "green" if bb_position < 0.2 else "orange"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>BB Position</h3>
                <p style="color: {bb_color}; font-size: 1.5rem; font-weight: bold;">
                    {bb_position:.2f}
                </p>
                <p>{bb_signal}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Stochastic
            stoch_signal = "Overbought" if current['STOCH_K'] > 80 else "Oversold" if current['STOCH_K'] < 20 else "Neutral"
            stoch_color = "red" if current['STOCH_K'] > 80 else "green" if current['STOCH_K'] < 20 else "orange"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Stochastic</h3>
                <p style="color: {stoch_color}; font-size: 1.5rem; font-weight: bold;">
                    {current['STOCH_K']:.1f}
                </p>
                <p>{stoch_signal}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Technical analysis chart
        st.subheader("📈 Technical Analysis Chart")
        
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxis=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f"{selected_stock} Price & Moving Averages",
                "RSI",
                "MACD",
                "Volume"
            ),
            row_width=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Price and Moving Averages
        fig.add_trace(go.Candlestick(
            x=processed_data.index,
            open=processed_data['Open'],
            high=processed_data['High'],
            low=processed_data['Low'],
            close=processed_data['Close'],
            name='Price'
        ), row=1, col=1)
        
        if 'SMA_20' in processed_data.columns:
            fig.add_trace(go.Scatter(
                x=processed_data.index,
                y=processed_data['SMA_20'],
                name='SMA 20',
                line=dict(color='orange', width=1)
            ), row=1, col=1)
        
        if 'SMA_50' in processed_data.columns:
            fig.add_trace(go.Scatter(
                x=processed_data.index,
                y=processed_data['SMA_50'],
                name='SMA 50',
                line=dict(color='blue', width=1)
            ), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(
            x=processed_data.index,
            y=processed_data['RSI'],
            name='RSI',
            line=dict(color='purple', width=2)
        ), row=2, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(
            x=processed_data.index,
            y=processed_data['MACD'],
            name='MACD',
            line=dict(color='blue', width=2)
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=processed_data.index,
            y=processed_data['MACD_signal'],
            name='MACD Signal',
            line=dict(color='red', width=2)
        ), row=3, col=1)
        
        # Volume
        fig.add_trace(go.Bar(
            x=processed_data.index,
            y=processed_data['Volume'],
            name='Volume',
            marker_color='lightblue'
        ), row=4, col=1)
        
        fig.update_layout(height=800, showlegend=True)
        fig.update_xaxes(rangeslider_visible=False)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def optimization_tab(self):
        """
        Portfolio optimization tab.
        """
        st.header("⚙️ Portfolio Optimization")
        
        # Get portfolio data
        portfolio = st.session_state.portfolio
        
        if not portfolio.positions:
            st.info("No positions in portfolio for optimization.")
            return
        
        # Fetch data for optimization
        symbols = list(portfolio.positions.keys())
        
        with st.spinner("Fetching data for optimization..."):
            price_data = {}
            for symbol in symbols:
                stock_data = self.data_collector.fetch_stock_data(symbol, period='1y')
                if stock_data:
                    price_data[symbol] = stock_data['data']['Close']
        
        if not price_data:
            st.error("Failed to fetch data for optimization.")
            return
        
        # Create price DataFrame
        price_df = pd.DataFrame(price_data)
        returns_df = price_df.pct_change().dropna()
        
        # Optimization methods
        st.subheader("🎯 Optimization Methods")
        
        optimization_method = st.selectbox(
            "Select Optimization Method",
            ["Mean-Variance", "Maximum Sharpe", "Minimum Variance", "Risk Parity", "Hierarchical Risk Parity", "Machine Learning"],
            index=0
        )
        
        if st.button("🚀 Run Optimization"):
            with st.spinner("Running optimization..."):
                if optimization_method == "Mean-Variance":
                    result = self.portfolio_optimizer.mean_variance_optimization(
                        returns_df.mean(), returns_df.cov()
                    )
                elif optimization_method == "Maximum Sharpe":
                    result = self.portfolio_optimizer.maximum_sharpe_optimization(
                        returns_df.mean(), returns_df.cov()
                    )
                elif optimization_method == "Minimum Variance":
                    result = self.portfolio_optimizer.minimum_variance_optimization(
                        returns_df.cov()
                    )
                elif optimization_method == "Risk Parity":
                    result = self.portfolio_optimizer.risk_parity_optimization(
                        returns_df.cov()
                    )
                elif optimization_method == "Hierarchical Risk Parity":
                    result = self.portfolio_optimizer.hierarchical_risk_parity(returns_df)
                elif optimization_method == "Machine Learning":
                    result = self.portfolio_optimizer.machine_learning_optimization(returns_df)
                
                st.session_state.optimization_result = result
        
        # Display optimization results
        if 'optimization_result' in st.session_state:
            result = st.session_state.optimization_result
            
            st.subheader("📊 Optimization Results")
            
            if result['optimization_success']:
                # Create results DataFrame
                results_data = []
                for i, symbol in enumerate(symbols):
                    results_data.append({
                        'Symbol': symbol,
                        'Current Weight': portfolio.positions[symbol] * price_df[symbol].iloc[-1] / portfolio.get_portfolio_value(price_df.iloc[-1].to_dict()),
                        'Optimal Weight': result['weights'][i],
                        'Weight Difference': result['weights'][i] - (portfolio.positions[symbol] * price_df[symbol].iloc[-1] / portfolio.get_portfolio_value(price_df.iloc[-1].to_dict()))
                    })
                
                results_df = pd.DataFrame(results_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(results_df, use_container_width=True)
                
                with col2:
                    # Optimization metrics
                    if 'expected_return' in result:
                        st.metric("Expected Return", f"{result['expected_return']:.2%}")
                    if 'volatility' in result:
                        st.metric("Portfolio Volatility", f"{result['volatility']:.2%}")
                    if 'sharpe_ratio' in result:
                        st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
                
                # Weight comparison chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=results_df['Symbol'],
                    y=results_df['Current Weight'],
                    name='Current Weight',
                    marker_color='lightblue'
                ))
                
                fig.add_trace(go.Bar(
                    x=results_df['Symbol'],
                    y=results_df['Optimal Weight'],
                    name='Optimal Weight',
                    marker_color='orange'
                ))
                
                fig.update_layout(
                    title="Current vs Optimal Weights",
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Rebalancing recommendations
                st.subheader("🔄 Rebalancing Recommendations")
                
                rebalance_threshold = st.slider(
                    "Rebalancing Threshold",
                    min_value=0.01,
                    max_value=0.20,
                    value=0.05,
                    step=0.01
                )
                
                current_weights = np.array([results_df['Current Weight'].iloc[i] for i in range(len(results_df))])
                optimal_weights = np.array([results_df['Optimal Weight'].iloc[i] for i in range(len(results_df))])
                
                rebalance_result = self.portfolio_optimizer.rebalance_portfolio(
                    current_weights, optimal_weights, rebalance_threshold
                )
                
                if rebalance_result['trades']:
                    st.write(f"**Rebalancing Trades Required:** {len(rebalance_result['trades'])}")
                    st.write(f"**Total Turnover:** {rebalance_result['total_turnover']:.2%}")
                    st.write(f"**Estimated Cost:** {format_currency(rebalance_result['rebalancing_cost'])}")
                    
                    trades_df = pd.DataFrame(rebalance_result['trades'])
                    st.dataframe(trades_df, use_container_width=True)
                else:
                    st.success("No rebalancing required!")
            else:
                st.error(f"Optimization failed: {result.get('error', 'Unknown error')}")
    
    def alerts_tab(self):
        """
        Alerts and notifications tab.
        """
        st.header("🚨 Alerts & Notifications")
        
        # Alert settings
        st.subheader("⚙️ Alert Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            price_change_threshold = st.slider(
                "Price Change Alert (%)",
                min_value=1,
                max_value=20,
                value=5,
                step=1
            )
            
            rsi_oversold = st.slider(
                "RSI Oversold Threshold",
                min_value=10,
                max_value=30,
                value=25,
                step=5
            )
        
        with col2:
            rsi_overbought = st.slider(
                "RSI Overbought Threshold",
                min_value=70,
                max_value=90,
                value=75,
                step=5
            )
            
            confidence_threshold = st.slider(
                "Signal Confidence Threshold",
                min_value=0.5,
                max_value=0.95,
                value=0.8,
                step=0.05
            )
        
        # Update alert criteria
        self.alert_system.alert_criteria.update({
            'price_change': price_change_threshold / 100,
            'rsi_extreme': {'oversold': rsi_oversold, 'overbought': rsi_overbought},
            'prediction_confidence': confidence_threshold
        })
        
        # Generate alerts
        if st.button("🔍 Check for Alerts"):
            selected_stock = st.session_state.get('selected_stock', 'RELIANCE.NS')
            
            # Fetch data
            stock_data = self.data_collector.fetch_stock_data(selected_stock, period='1d')
            if stock_data:
                processed_data = self.data_collector.calculate_technical_indicators(stock_data['data'])
                
                # Simulate signal data
                signal_data = {
                    'symbol': selected_stock,
                    'price_change_pct': 0.06,  # 6% change
                    'confidence': 0.85,
                    'technical_indicators': {
                        'RSI': 25,
                        'MACD': 0.5,
                        'MACD_Signal': 0.3,
                        'SMA_20': 2500,
                        'SMA_50': 2450,
                        'BB_Position': 0.1,
                        'Volume_Ratio': 2.0
                    }
                }
                
                alerts = self.alert_system.check_alerts(signal_data, processed_data)
                st.session_state.current_alerts = alerts
        
        # Display current alerts
        if 'current_alerts' in st.session_state and st.session_state.current_alerts:
            st.subheader("🚨 Current Alerts")
            
            for alert in st.session_state.current_alerts:
                alert_class = {
                    'HIGH': 'alert-high',
                    'MEDIUM': 'alert-medium',
                    'LOW': 'alert-low'
                }.get(alert['priority'], 'alert-low')
                
                st.markdown(f"""
                <div class="metric-card {alert_class}">
                    <h4>{alert['type']}</h4>
                    <p><strong>Priority:</strong> {alert['priority']}</p>
                    <p>{alert['message']}</p>
                    <p><small>{alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</small></p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No current alerts.")
        
        # Alert history
        st.subheader("📋 Alert History")
        
        recent_alerts = self.alert_system.get_recent_alerts(24)
        
        if recent_alerts:
            alert_summary = self.alert_system.get_alert_summary(24)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Alerts (24h)", alert_summary['total_alerts'])
            
            with col2:
                st.metric("High Priority", alert_summary['high_priority'])
            
            with col3:
                st.metric("Medium Priority", alert_summary['medium_priority'])
            
            # Alert history table
            alerts_df = pd.DataFrame(recent_alerts[-10:])  # Last 10 alerts
            if not alerts_df.empty:
                st.dataframe(alerts_df[['timestamp', 'type', 'priority', 'message']], use_container_width=True)
        else:
            st.info("No alert history available.")

def main():
    """
    Main function to run the Streamlit app.
    """
    app = StreamlitTradingApp()
    app.run()

if __name__ == "__main__":
    main() 