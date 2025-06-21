"""
sentiment.py
Sentiment analysis for market sentiment from multiple sources.
"""
import pandas as pd
import numpy as np
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import json

class SentimentAnalyzer:
    """
    Advanced sentiment analysis for financial markets.
    Analyzes news, social media, and financial data for market sentiment.
    """
    
    def __init__(self):
        self.sentiment_cache = {}
        self.news_sources = [
            'moneycontrol.com',
            'economictimes.indiatimes.com',
            'livemint.com',
            'ndtv.com/business',
            'cnbctv18.com'
        ]
        self.sentiment_keywords = {
            'positive': ['bullish', 'surge', 'rally', 'gain', 'profit', 'growth', 'positive', 'up', 'higher'],
            'negative': ['bearish', 'fall', 'drop', 'loss', 'decline', 'negative', 'down', 'lower', 'crash'],
            'neutral': ['stable', 'steady', 'unchanged', 'flat', 'consolidate']
        }
    
    def analyze_text_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of text using TextBlob and custom keywords.
        """
        if not text or len(text.strip()) < 10:
            return {'polarity': 0, 'subjectivity': 0, 'sentiment': 'neutral'}
        
        # TextBlob analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Custom keyword analysis
        text_lower = text.lower()
        positive_count = sum(1 for word in self.sentiment_keywords['positive'] if word in text_lower)
        negative_count = sum(1 for word in self.sentiment_keywords['negative'] if word in text_lower)
        neutral_count = sum(1 for word in self.sentiment_keywords['neutral'] if word in text_lower)
        
        # Combine TextBlob and keyword analysis
        keyword_score = (positive_count - negative_count) / max(1, positive_count + negative_count + neutral_count)
        combined_polarity = (polarity + keyword_score) / 2
        
        # Determine sentiment category
        if combined_polarity > 0.1:
            sentiment = 'positive'
        elif combined_polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'polarity': combined_polarity,
            'subjectivity': subjectivity,
            'sentiment': sentiment,
            'textblob_polarity': polarity,
            'keyword_score': keyword_score,
            'positive_keywords': positive_count,
            'negative_keywords': negative_count,
            'neutral_keywords': neutral_count
        }
    
    def fetch_news_headlines(self, symbol: str, days: int = 7) -> List[Dict]:
        """
        Fetch news headlines related to a stock symbol.
        """
        headlines = []
        
        # Search terms for the stock
        search_terms = [
            symbol.replace('.NS', ''),
            symbol.replace('.NS', ' stock'),
            symbol.replace('.NS', ' share price'),
            symbol.replace('.NS', ' company')
        ]
        
        for term in search_terms:
            try:
                # Simulate news fetching (in real implementation, use news APIs)
                # This is a placeholder for actual news API integration
                mock_headlines = [
                    f"{term} reports strong quarterly results",
                    f"{term} announces new product launch",
                    f"{term} faces regulatory challenges",
                    f"{term} expands into new markets",
                    f"{term} stock price reaches new high"
                ]
                
                for headline in mock_headlines:
                    headlines.append({
                        'title': headline,
                        'source': 'mock_news',
                        'date': datetime.now() - timedelta(days=np.random.randint(0, days)),
                        'url': f"https://example.com/news/{hash(headline)}"
                    })
                
            except Exception as e:
                print(f"Error fetching news for {term}: {e}")
        
        return headlines
    
    def analyze_stock_sentiment(self, symbol: str, days: int = 7) -> Dict:
        """
        Analyze sentiment for a specific stock.
        """
        # Check cache
        cache_key = f"{symbol}_{days}"
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        # Fetch news headlines
        headlines = self.fetch_news_headlines(symbol, days)
        
        if not headlines:
            return {
                'symbol': symbol,
                'overall_sentiment': 'neutral',
                'sentiment_score': 0,
                'confidence': 0,
                'headlines_analyzed': 0,
                'sentiment_breakdown': {'positive': 0, 'negative': 0, 'neutral': 0},
                'timestamp': datetime.now()
            }
        
        # Analyze each headline
        sentiments = []
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for headline in headlines:
            sentiment = self.analyze_text_sentiment(headline['title'])
            sentiments.append(sentiment)
            sentiment_counts[sentiment['sentiment']] += 1
        
        # Calculate overall sentiment
        if sentiments:
            avg_polarity = np.mean([s['polarity'] for s in sentiments])
            avg_subjectivity = np.mean([s['subjectivity'] for s in sentiments])
            
            # Determine overall sentiment
            if avg_polarity > 0.1:
                overall_sentiment = 'positive'
            elif avg_polarity < -0.1:
                overall_sentiment = 'negative'
            else:
                overall_sentiment = 'neutral'
            
            # Calculate confidence based on subjectivity and number of headlines
            confidence = min(avg_subjectivity * len(headlines) / 10, 1.0)
        else:
            avg_polarity = 0
            avg_subjectivity = 0
            overall_sentiment = 'neutral'
            confidence = 0
        
        result = {
            'symbol': symbol,
            'overall_sentiment': overall_sentiment,
            'sentiment_score': avg_polarity,
            'confidence': confidence,
            'headlines_analyzed': len(headlines),
            'sentiment_breakdown': sentiment_counts,
            'avg_subjectivity': avg_subjectivity,
            'headlines': headlines[:5],  # Top 5 headlines
            'timestamp': datetime.now()
        }
        
        # Cache result
        self.sentiment_cache[cache_key] = result
        return result
    
    def get_market_sentiment(self, symbols: List[str]) -> Dict:
        """
        Get overall market sentiment from multiple stocks.
        """
        market_sentiments = []
        
        for symbol in symbols:
            sentiment = self.analyze_stock_sentiment(symbol, days=3)
            market_sentiments.append(sentiment)
        
        if not market_sentiments:
            return {
                'market_sentiment': 'neutral',
                'sentiment_score': 0,
                'confidence': 0,
                'stocks_analyzed': 0,
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0}
            }
        
        # Calculate market-wide sentiment
        sentiment_scores = [s['sentiment_score'] for s in market_sentiments]
        avg_market_sentiment = np.mean(sentiment_scores)
        
        # Count sentiment distribution
        sentiment_distribution = {'positive': 0, 'negative': 0, 'neutral': 0}
        for sentiment in market_sentiments:
            sentiment_distribution[sentiment['overall_sentiment']] += 1
        
        # Determine overall market sentiment
        if avg_market_sentiment > 0.1:
            market_sentiment = 'positive'
        elif avg_market_sentiment < -0.1:
            market_sentiment = 'negative'
        else:
            market_sentiment = 'neutral'
        
        # Calculate confidence
        confidences = [s['confidence'] for s in market_sentiments]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return {
            'market_sentiment': market_sentiment,
            'sentiment_score': avg_market_sentiment,
            'confidence': avg_confidence,
            'stocks_analyzed': len(market_sentiments),
            'sentiment_distribution': sentiment_distribution,
            'individual_sentiments': market_sentiments,
            'timestamp': datetime.now()
        }
    
    def analyze_social_media_sentiment(self, symbol: str) -> Dict:
        """
        Analyze social media sentiment for a stock.
        Note: This is a placeholder for actual social media API integration.
        """
        # Simulate social media sentiment analysis
        mock_tweets = [
            f"$RELIANCE looking bullish today! #stocks #trading",
            f"Not sure about {symbol} performance lately",
            f"Great earnings report from {symbol}",
            f"{symbol} stock price prediction for next week",
            f"Bearish on {symbol} due to market conditions"
        ]
        
        sentiments = []
        for tweet in mock_tweets:
            sentiment = self.analyze_text_sentiment(tweet)
            sentiments.append(sentiment)
        
        if sentiments:
            avg_polarity = np.mean([s['polarity'] for s in sentiments])
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            for s in sentiments:
                sentiment_counts[s['sentiment']] += 1
            
            return {
                'symbol': symbol,
                'social_sentiment': 'positive' if avg_polarity > 0.1 else 'negative' if avg_polarity < -0.1 else 'neutral',
                'sentiment_score': avg_polarity,
                'posts_analyzed': len(sentiments),
                'sentiment_breakdown': sentiment_counts,
                'timestamp': datetime.now()
            }
        
        return {
            'symbol': symbol,
            'social_sentiment': 'neutral',
            'sentiment_score': 0,
            'posts_analyzed': 0,
            'sentiment_breakdown': {'positive': 0, 'negative': 0, 'neutral': 0},
            'timestamp': datetime.now()
        }
    
    def calculate_fear_greed_index(self, market_data: Dict) -> Dict:
        """
        Calculate fear and greed index based on market indicators.
        """
        # Extract market indicators
        volatility = market_data.get('volatility', 0.02)
        momentum = market_data.get('momentum', 0)
        market_volume = market_data.get('volume_ratio', 1.0)
        put_call_ratio = market_data.get('put_call_ratio', 1.0)
        junk_bond_demand = market_data.get('junk_bond_demand', 0.5)
        market_momentum = market_data.get('market_momentum', 0)
        
        # Calculate individual components (0-100 scale)
        volatility_score = max(0, min(100, (1 - volatility * 50)))
        momentum_score = max(0, min(100, 50 + momentum * 100))
        volume_score = max(0, min(100, 50 + (market_volume - 1) * 50))
        put_call_score = max(0, min(100, 50 + (1 - put_call_ratio) * 50))
        junk_bond_score = max(0, min(100, junk_bond_demand * 100))
        market_momentum_score = max(0, min(100, 50 + market_momentum * 100))
        
        # Calculate average fear and greed index
        fear_greed_index = np.mean([
            volatility_score, momentum_score, volume_score,
            put_call_score, junk_bond_score, market_momentum_score
        ])
        
        # Determine sentiment category
        if fear_greed_index >= 75:
            sentiment = 'Extreme Greed'
        elif fear_greed_index >= 60:
            sentiment = 'Greed'
        elif fear_greed_index >= 40:
            sentiment = 'Neutral'
        elif fear_greed_index >= 25:
            sentiment = 'Fear'
        else:
            sentiment = 'Extreme Fear'
        
        return {
            'fear_greed_index': fear_greed_index,
            'sentiment': sentiment,
            'components': {
                'volatility': volatility_score,
                'momentum': momentum_score,
                'volume': volume_score,
                'put_call_ratio': put_call_score,
                'junk_bond_demand': junk_bond_score,
                'market_momentum': market_momentum_score
            },
            'timestamp': datetime.now()
        }
    
    def integrate_sentiment_with_trading(self, symbol: str, technical_signals: Dict) -> Dict:
        """
        Integrate sentiment analysis with technical trading signals.
        """
        # Get sentiment data
        news_sentiment = self.analyze_stock_sentiment(symbol, days=3)
        social_sentiment = self.analyze_social_media_sentiment(symbol)
        
        # Calculate sentiment-adjusted signal
        sentiment_score = (news_sentiment['sentiment_score'] + social_sentiment['sentiment_score']) / 2
        
        # Adjust technical signal based on sentiment
        original_signal = technical_signals.get('signal', 'HOLD')
        original_confidence = technical_signals.get('confidence', 0.5)
        
        # Sentiment adjustment factor
        sentiment_adjustment = sentiment_score * 0.2  # 20% weight to sentiment
        
        # Adjust confidence based on sentiment alignment
        if original_signal == 'BUY' and sentiment_score > 0.1:
            adjusted_confidence = original_confidence + sentiment_adjustment
        elif original_signal == 'SELL' and sentiment_score < -0.1:
            adjusted_confidence = original_confidence + abs(sentiment_adjustment)
        else:
            adjusted_confidence = original_confidence - abs(sentiment_adjustment)
        
        # Ensure confidence is within bounds
        adjusted_confidence = max(0, min(1, adjusted_confidence))
        
        return {
            'original_signal': original_signal,
            'original_confidence': original_confidence,
            'sentiment_score': sentiment_score,
            'news_sentiment': news_sentiment['overall_sentiment'],
            'social_sentiment': social_sentiment['social_sentiment'],
            'adjusted_confidence': adjusted_confidence,
            'sentiment_adjustment': sentiment_adjustment,
            'final_signal': original_signal if adjusted_confidence > 0.6 else 'HOLD',
            'sentiment_data': {
                'news': news_sentiment,
                'social': social_sentiment
            }
        }
    
    def clear_cache(self):
        """Clear sentiment cache."""
        self.sentiment_cache.clear()
    
    def get_sentiment_summary(self, symbols: List[str]) -> Dict:
        """
        Get comprehensive sentiment summary for multiple symbols.
        """
        summary = {
            'total_symbols': len(symbols),
            'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
            'avg_sentiment_score': 0,
            'top_positive': [],
            'top_negative': [],
            'timestamp': datetime.now()
        }
        
        symbol_sentiments = []
        
        for symbol in symbols:
            sentiment = self.analyze_stock_sentiment(symbol, days=3)
            symbol_sentiments.append({
                'symbol': symbol,
                'sentiment': sentiment['overall_sentiment'],
                'score': sentiment['sentiment_score'],
                'confidence': sentiment['confidence']
            })
            summary['sentiment_distribution'][sentiment['overall_sentiment']] += 1
        
        if symbol_sentiments:
            summary['avg_sentiment_score'] = np.mean([s['score'] for s in symbol_sentiments])
            
            # Get top positive and negative
            positive_symbols = [s for s in symbol_sentiments if s['sentiment'] == 'positive']
            negative_symbols = [s for s in symbol_sentiments if s['sentiment'] == 'negative']
            
            summary['top_positive'] = sorted(positive_symbols, key=lambda x: x['score'], reverse=True)[:5]
            summary['top_negative'] = sorted(negative_symbols, key=lambda x: x['score'])[:5]
        
        return summary 