"""
alerts.py
Real-time alert system for trading signals and risk management.
"""
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd

class AlertSystem:
    """
    Real-time alert system for trading signals and market conditions.
    Provides configurable alerts for price movements, technical indicators, and ML predictions.
    """
    
    def __init__(self):
        self.alerts = []
        self.alert_criteria = {
            'price_change': 0.05,  # 5% price change
            'volume_spike': 2.0,   # 2x average volume
            'rsi_extreme': {'oversold': 25, 'overbought': 75},
            'prediction_confidence': 0.8,
            'macd_crossover': True,
            'bollinger_breakout': True,
            'support_resistance': 0.02  # 2% from support/resistance
        }
        self.alert_types = {
            'PRICE_MOVEMENT': 'Price movement alert',
            'RSI_OVERSOLD': 'RSI oversold condition',
            'RSI_OVERBOUGHT': 'RSI overbought condition',
            'HIGH_CONFIDENCE_SIGNAL': 'High confidence trading signal',
            'VOLUME_SPIKE': 'Unusual volume activity',
            'MACD_CROSSOVER': 'MACD crossover signal',
            'BOLLINGER_BREAKOUT': 'Bollinger Bands breakout',
            'SUPPORT_RESISTANCE': 'Support/Resistance level reached',
            'RISK_LIMIT': 'Risk management limit reached',
            'PORTFOLIO_ALERT': 'Portfolio performance alert'
        }
    
    def check_alerts(self, signal_data: Dict, current_data: pd.DataFrame) -> List[Dict]:
        """
        Check for alert conditions based on signal data and current market data.
        Returns list of triggered alerts.
        """
        alerts = []
        
        # Price change alert
        if abs(signal_data['price_change_pct']) > self.alert_criteria['price_change']:
            direction = "UP" if signal_data['price_change_pct'] > 0 else "DOWN"
            alerts.append({
                'type': 'PRICE_MOVEMENT',
                'symbol': signal_data['symbol'],
                'message': f"Price predicted to move {direction} by {abs(signal_data['price_change_pct']):.2%}",
                'priority': 'HIGH' if abs(signal_data['price_change_pct']) > 0.1 else 'MEDIUM',
                'timestamp': datetime.now(),
                'value': signal_data['price_change_pct']
            })
        
        # RSI extreme alert
        rsi = signal_data['technical_indicators']['RSI']
        if rsi < self.alert_criteria['rsi_extreme']['oversold']:
            alerts.append({
                'type': 'RSI_OVERSOLD',
                'symbol': signal_data['symbol'],
                'message': f"RSI extremely oversold at {rsi:.1f}",
                'priority': 'HIGH',
                'timestamp': datetime.now(),
                'value': rsi
            })
        elif rsi > self.alert_criteria['rsi_extreme']['overbought']:
            alerts.append({
                'type': 'RSI_OVERBOUGHT',
                'symbol': signal_data['symbol'],
                'message': f"RSI extremely overbought at {rsi:.1f}",
                'priority': 'HIGH',
                'timestamp': datetime.now(),
                'value': rsi
            })
        
        # High confidence signal alert
        if signal_data['confidence'] > self.alert_criteria['prediction_confidence']:
            alerts.append({
                'type': 'HIGH_CONFIDENCE_SIGNAL',
                'symbol': signal_data['symbol'],
                'message': f"High confidence {signal_data['signal']} signal ({signal_data['confidence']:.2%})",
                'priority': 'HIGH',
                'timestamp': datetime.now(),
                'value': signal_data['confidence']
            })
        
        # Volume spike alert
        volume_ratio = signal_data['technical_indicators'].get('Volume_Ratio', 1.0)
        if volume_ratio > self.alert_criteria['volume_spike']:
            alerts.append({
                'type': 'VOLUME_SPIKE',
                'symbol': signal_data['symbol'],
                'message': f"Volume spike detected: {volume_ratio:.1f}x average volume",
                'priority': 'MEDIUM',
                'timestamp': datetime.now(),
                'value': volume_ratio
            })
        
        # MACD crossover alert
        if self.alert_criteria['macd_crossover']:
            macd = signal_data['technical_indicators']['MACD']
            macd_signal = signal_data['technical_indicators']['MACD_Signal']
            if abs(macd - macd_signal) / abs(macd_signal) < 0.01:  # Within 1%
                crossover_type = "bullish" if macd > macd_signal else "bearish"
                alerts.append({
                    'type': 'MACD_CROSSOVER',
                    'symbol': signal_data['symbol'],
                    'message': f"MACD {crossover_type} crossover detected",
                    'priority': 'MEDIUM',
                    'timestamp': datetime.now(),
                    'value': macd - macd_signal
                })
        
        # Bollinger Bands breakout alert
        if self.alert_criteria['bollinger_breakout']:
            bb_position = signal_data['technical_indicators']['BB_Position']
            if bb_position > 0.95 or bb_position < 0.05:
                breakout_type = "upper" if bb_position > 0.95 else "lower"
                alerts.append({
                    'type': 'BOLLINGER_BREAKOUT',
                    'symbol': signal_data['symbol'],
                    'message': f"Bollinger Bands {breakout_type} breakout detected",
                    'priority': 'MEDIUM',
                    'timestamp': datetime.now(),
                    'value': bb_position
                })
        
        # Risk-reward ratio alert
        if signal_data.get('risk_reward_ratio', 0) > 3.0:
            alerts.append({
                'type': 'RISK_REWARD',
                'symbol': signal_data['symbol'],
                'message': f"Excellent risk-reward ratio: {signal_data['risk_reward_ratio']:.1f}",
                'priority': 'HIGH',
                'timestamp': datetime.now(),
                'value': signal_data['risk_reward_ratio']
            })
        
        self.alerts.extend(alerts)
        return alerts
    
    def get_recent_alerts(self, hours: int = 24, symbol: str = None) -> List[Dict]:
        """
        Get alerts from the last N hours, optionally filtered by symbol.
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]
        
        if symbol:
            recent_alerts = [alert for alert in recent_alerts if alert['symbol'] == symbol]
        
        return recent_alerts
    
    def get_alerts_by_type(self, alert_type: str, hours: int = 24) -> List[Dict]:
        """
        Get alerts of a specific type from the last N hours.
        """
        recent_alerts = self.get_recent_alerts(hours)
        return [alert for alert in recent_alerts if alert['type'] == alert_type]
    
    def get_high_priority_alerts(self, hours: int = 24) -> List[Dict]:
        """
        Get high priority alerts from the last N hours.
        """
        recent_alerts = self.get_recent_alerts(hours)
        return [alert for alert in recent_alerts if alert['priority'] == 'HIGH']
    
    def clear_old_alerts(self, days: int = 7):
        """
        Clear alerts older than specified days to manage memory.
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        self.alerts = [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]
    
    def get_alert_summary(self, hours: int = 24) -> Dict:
        """
        Get summary statistics of recent alerts.
        """
        recent_alerts = self.get_recent_alerts(hours)
        
        if not recent_alerts:
            return {
                'total_alerts': 0,
                'high_priority': 0,
                'medium_priority': 0,
                'low_priority': 0,
                'by_type': {},
                'by_symbol': {}
            }
        
        # Count by priority
        high_priority = len([a for a in recent_alerts if a['priority'] == 'HIGH'])
        medium_priority = len([a for a in recent_alerts if a['priority'] == 'MEDIUM'])
        low_priority = len([a for a in recent_alerts if a['priority'] == 'LOW'])
        
        # Count by type
        by_type = {}
        for alert in recent_alerts:
            alert_type = alert['type']
            by_type[alert_type] = by_type.get(alert_type, 0) + 1
        
        # Count by symbol
        by_symbol = {}
        for alert in recent_alerts:
            symbol = alert['symbol']
            by_symbol[symbol] = by_symbol.get(symbol, 0) + 1
        
        return {
            'total_alerts': len(recent_alerts),
            'high_priority': high_priority,
            'medium_priority': medium_priority,
            'low_priority': low_priority,
            'by_type': by_type,
            'by_symbol': by_symbol
        }
    
    def add_custom_alert(self, alert_type: str, symbol: str, message: str, priority: str = 'MEDIUM', value: float = None):
        """
        Add a custom alert manually.
        """
        alert = {
            'type': alert_type,
            'symbol': symbol,
            'message': message,
            'priority': priority,
            'timestamp': datetime.now(),
            'value': value
        }
        self.alerts.append(alert)
    
    def check_portfolio_alerts(self, portfolio_metrics: Dict) -> List[Dict]:
        """
        Check for portfolio-level alerts based on performance metrics.
        """
        alerts = []
        
        # Drawdown alert
        max_drawdown = portfolio_metrics.get('Max_Drawdown', 0)
        if max_drawdown > 0.15:  # 15% drawdown
            alerts.append({
                'type': 'PORTFOLIO_ALERT',
                'symbol': 'PORTFOLIO',
                'message': f"High drawdown detected: {max_drawdown:.2%}",
                'priority': 'HIGH',
                'timestamp': datetime.now(),
                'value': max_drawdown
            })
        
        # Sharpe ratio alert
        sharpe_ratio = portfolio_metrics.get('Sharpe_Ratio', 0)
        if sharpe_ratio < 0.5:
            alerts.append({
                'type': 'PORTFOLIO_ALERT',
                'symbol': 'PORTFOLIO',
                'message': f"Low Sharpe ratio: {sharpe_ratio:.2f}",
                'priority': 'MEDIUM',
                'timestamp': datetime.now(),
                'value': sharpe_ratio
            })
        
        # Return alert
        total_return = portfolio_metrics.get('Total_Return', 0)
        if total_return < -0.1:  # -10% return
            alerts.append({
                'type': 'PORTFOLIO_ALERT',
                'symbol': 'PORTFOLIO',
                'message': f"Negative returns: {total_return:.2%}",
                'priority': 'HIGH',
                'timestamp': datetime.now(),
                'value': total_return
            })
        
        self.alerts.extend(alerts)
        return alerts 