"""
dashboard.py
Advanced visualization dashboard for the trading system.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class TradingDashboard:
    """
    Advanced visualization dashboard with interactive charts and real-time updates.
    """
    
    def __init__(self):
        self.figures = {}
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
    
    def plot_price_prediction(self, actual_prices: np.ndarray, predicted_prices: np.ndarray, symbol: str) -> go.Figure:
        """
        Create interactive price prediction chart with actual vs predicted prices.
        """
        fig = go.Figure()
        
        # Generate dates for x-axis
        dates = pd.date_range(start='2023-01-01', periods=len(actual_prices), freq='D')
        
        # Actual prices
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual_prices.flatten(),
            name='Actual Prices',
            line=dict(color=self.color_scheme['primary'], width=2),
            mode='lines'
        ))
        
        # Predicted prices
        fig.add_trace(go.Scatter(
            x=dates,
            y=predicted_prices.flatten(),
            name='Predicted Prices',
            line=dict(color=self.color_scheme['danger'], width=2, dash='dash'),
            mode='lines'
        ))
        
        # Add confidence interval (simulated)
        confidence_upper = predicted_prices.flatten() * 1.02
        confidence_lower = predicted_prices.flatten() * 0.98
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=confidence_upper,
            name='Confidence Upper',
            line=dict(color='rgba(255,0,0,0.3)', width=1),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=confidence_lower,
            name='Confidence Lower',
            line=dict(color='rgba(255,0,0,0.3)', width=1),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)',
            showlegend=False
        ))
        
        fig.update_layout(
            title=f'{symbol} - Actual vs Predicted Prices',
            xaxis_title='Date',
            yaxis_title='Price (₹)',
            hovermode='x unified',
            template='plotly_dark',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_technical_indicators(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Create comprehensive technical analysis chart with multiple indicators.
        """
        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxis=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f'{symbol} Price & Moving Averages',
                'RSI',
                'MACD',
                'Bollinger Bands',
                'Volume'
            ),
            row_width=[0.3, 0.15, 0.15, 0.15, 0.25]
        )
        
        # Price and Moving Averages
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color=self.color_scheme['success'],
            decreasing_line_color=self.color_scheme['danger']
        ), row=1, col=1)
        
        # Moving averages
        if 'SMA_20' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA_20'],
                name='SMA 20',
                line=dict(color=self.color_scheme['secondary'], width=1)
            ), row=1, col=1)
        
        if 'SMA_50' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA_50'],
                name='SMA 50',
                line=dict(color=self.color_scheme['info'], width=1)
            ), row=1, col=1)
        
        # Bollinger Bands
        if all(col in data.columns for col in ['BB_upper', 'BB_middle', 'BB_lower']):
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['BB_upper'],
                name='BB Upper',
                line=dict(color='rgba(128,128,128,0.8)', width=1, dash='dash'),
                showlegend=False
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['BB_lower'],
                name='BB Lower',
                line=dict(color='rgba(128,128,128,0.8)', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                showlegend=False
            ), row=1, col=1)
        
        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['RSI'],
                name='RSI',
                line=dict(color=self.color_scheme['warning'], width=2)
            ), row=2, col=1)
            
            # RSI overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
        
        # MACD
        if all(col in data.columns for col in ['MACD', 'MACD_signal', 'MACD_hist']):
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['MACD'],
                name='MACD',
                line=dict(color=self.color_scheme['primary'], width=2)
            ), row=3, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['MACD_signal'],
                name='MACD Signal',
                line=dict(color=self.color_scheme['danger'], width=2)
            ), row=3, col=1)
            
            fig.add_trace(go.Bar(
                x=data.index,
                y=data['MACD_hist'],
                name='MACD Histogram',
                marker_color=np.where(data['MACD_hist'] >= 0, self.color_scheme['success'], self.color_scheme['danger'])
            ), row=3, col=1)
        
        # Bollinger Bands as separate subplot
        if all(col in data.columns for col in ['BB_upper', 'BB_middle', 'BB_lower']):
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                name='Price',
                line=dict(color=self.color_scheme['primary'], width=2)
            ), row=4, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['BB_upper'],
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash')
            ), row=4, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['BB_middle'],
                name='BB Middle',
                line=dict(color='gray', width=1)
            ), row=4, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['BB_lower'],
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)'
            ), row=4, col=1)
        
        # Volume
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=self.color_scheme['info'],
            opacity=0.7
        ), row=5, col=1)
        
        fig.update_layout(
            title=f'{symbol} - Technical Analysis Dashboard',
            xaxis_title='Date',
            template='plotly_dark',
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig
    
    def plot_portfolio_performance(self, portfolio_results: Dict) -> go.Figure:
        """
        Create comprehensive portfolio performance dashboard.
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Portfolio Value Over Time',
                'Trade Distribution',
                'Monthly Returns',
                'Risk-Return Analysis'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Portfolio value over time (simulated data)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        portfolio_values = np.cumsum(np.random.normal(0, 0.01, 100)) * 10000 + 100000
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            name='Portfolio Value',
            line=dict(color=self.color_scheme['success'], width=3)
        ), row=1, col=1)
        
        # Trade distribution
        executed_trades = [t for t in portfolio_results.get('trades', []) if t.get('status') == 'EXECUTED']
        if executed_trades:
            trade_symbols = [t['symbol'] for t in executed_trades]
            symbol_counts = pd.Series(trade_symbols).value_counts()
            
            fig.add_trace(go.Bar(
                x=symbol_counts.index,
                y=symbol_counts.values,
                name='Trades per Symbol',
                marker_color=self.color_scheme['primary']
            ), row=1, col=2)
        
        # Monthly returns (simulated)
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        returns = np.random.normal(0.02, 0.05, len(months))
        colors = [self.color_scheme['success'] if r > 0 else self.color_scheme['danger'] for r in returns]
        
        fig.add_trace(go.Bar(
            x=months,
            y=returns,
            name='Monthly Returns',
            marker_color=colors
        ), row=2, col=1)
        
        # Risk-Return scatter (simulated)
        symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
        returns_data = np.random.normal(0.12, 0.05, len(symbols))
        volatility_data = np.random.normal(0.25, 0.1, len(symbols))
        
        fig.add_trace(go.Scatter(
            x=volatility_data,
            y=returns_data,
            mode='markers+text',
            text=symbols,
            textposition='top center',
            name='Risk-Return',
            marker=dict(size=15, color=self.color_scheme['warning'])
        ), row=2, col=2)
        
        fig.update_layout(
            title='Portfolio Performance Dashboard',
            template='plotly_dark',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def plot_model_training_history(self, history) -> go.Figure:
        """
        Create model training progress visualization.
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Training Loss', 'Training Metrics')
        )
        
        epochs = range(1, len(history.history['loss']) + 1)
        
        # Loss
        fig.add_trace(go.Scatter(
            x=list(epochs),
            y=history.history['loss'],
            name='Training Loss',
            line=dict(color=self.color_scheme['danger'], width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=list(epochs),
            y=history.history['val_loss'],
            name='Validation Loss',
            line=dict(color=self.color_scheme['primary'], width=2)
        ), row=1, col=1)
        
        # MAE
        if 'mae' in history.history:
            fig.add_trace(go.Scatter(
                x=list(epochs),
                y=history.history['mae'],
                name='Training MAE',
                line=dict(color=self.color_scheme['success'], width=2)
            ), row=1, col=2)
            
            fig.add_trace(go.Scatter(
                x=list(epochs),
                y=history.history['val_mae'],
                name='Validation MAE',
                line=dict(color=self.color_scheme['warning'], width=2)
            ), row=1, col=2)
        
        fig.update_layout(
            title='Model Training Progress',
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    def plot_signal_analysis(self, signals: List[Dict]) -> go.Figure:
        """
        Create signal analysis and performance visualization.
        """
        if not signals:
            return go.Figure()
        
        df_signals = pd.DataFrame(signals)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Signal Confidence Distribution',
                'Signal Types',
                'Price Change vs Confidence',
                'Signal Timeline'
            )
        )
        
        # Signal confidence distribution
        fig.add_trace(go.Histogram(
            x=df_signals['confidence'],
            name='Confidence Distribution',
            nbinsx=20,
            marker_color=self.color_scheme['primary']
        ), row=1, col=1)
        
        # Signal types
        signal_counts = df_signals['signal'].value_counts()
        fig.add_trace(go.Pie(
            labels=signal_counts.index,
            values=signal_counts.values,
            name='Signal Types'
        ), row=1, col=2)
        
        # Price change vs confidence
        fig.add_trace(go.Scatter(
            x=df_signals['confidence'],
            y=df_signals['price_change_pct'],
            mode='markers',
            name='Price Change vs Confidence',
            marker=dict(
                size=10,
                color=df_signals['price_change_pct'],
                colorscale='RdYlGn',
                showscale=True
            )
        ), row=2, col=1)
        
        # Signal timeline
        fig.add_trace(go.Scatter(
            x=df_signals['timestamp'],
            y=df_signals['confidence'],
            mode='lines+markers',
            name='Signal Confidence Over Time',
            line=dict(color=self.color_scheme['info'], width=2)
        ), row=2, col=2)
        
        fig.update_layout(
            title='Signal Analysis Dashboard',
            template='plotly_dark',
            height=600
        )
        
        return fig 