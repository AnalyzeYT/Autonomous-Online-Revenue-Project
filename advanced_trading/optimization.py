"""
optimization.py
Portfolio optimization using Modern Portfolio Theory and machine learning.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """
    Advanced portfolio optimization using multiple strategies.
    Implements Modern Portfolio Theory, Risk Parity, and ML-based optimization.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.optimization_results = {}
        
    def calculate_returns_and_covariance(self, price_data: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Calculate returns and covariance matrix from price data.
        """
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # Calculate covariance matrix
        covariance_matrix = returns.cov()
        
        return returns, covariance_matrix
    
    def mean_variance_optimization(self, returns: pd.Series, covariance_matrix: pd.DataFrame, 
                                 target_return: Optional[float] = None, 
                                 risk_tolerance: float = 1.0) -> Dict:
        """
        Mean-Variance Optimization (Markowitz).
        """
        n_assets = len(returns)
        
        # Expected returns
        expected_returns = returns.mean()
        
        # Objective function: minimize portfolio variance
        def objective(weights):
            portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
            return portfolio_variance
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq', 
                'fun': lambda x: np.sum(expected_returns * x) - target_return
            })
        
        # Bounds: no short selling
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            objective, 
            initial_weights, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_return = np.sum(expected_returns * optimal_weights)
            portfolio_volatility = np.sqrt(result.fun)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'optimization_success': True
            }
        else:
            return {
                'weights': np.array([1/n_assets] * n_assets),
                'expected_return': np.mean(expected_returns),
                'volatility': np.sqrt(np.mean(np.diag(covariance_matrix))),
                'sharpe_ratio': 0,
                'optimization_success': False,
                'error': result.message
            }
    
    def risk_parity_optimization(self, covariance_matrix: pd.DataFrame) -> Dict:
        """
        Risk Parity optimization - equal risk contribution from each asset.
        """
        n_assets = len(covariance_matrix)
        
        def risk_contribution(weights):
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            asset_contribution = weights * (np.dot(covariance_matrix, weights)) / portfolio_volatility
            return asset_contribution
        
        def objective(weights):
            asset_contribution = risk_contribution(weights)
            target_contribution = 1.0 / n_assets
            return np.sum((asset_contribution - target_contribution) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        # Bounds: no short selling
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights)))
            
            return {
                'weights': optimal_weights,
                'portfolio_volatility': portfolio_volatility,
                'risk_contributions': risk_contribution(optimal_weights),
                'optimization_success': True
            }
        else:
            return {
                'weights': np.array([1/n_assets] * n_assets),
                'portfolio_volatility': np.sqrt(np.mean(np.diag(covariance_matrix))),
                'risk_contributions': np.array([1/n_assets] * n_assets),
                'optimization_success': False,
                'error': result.message
            }
    
    def maximum_sharpe_optimization(self, returns: pd.Series, covariance_matrix: pd.DataFrame) -> Dict:
        """
        Maximum Sharpe Ratio optimization.
        """
        n_assets = len(returns)
        expected_returns = returns.mean()
        
        def negative_sharpe(weights):
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            return -sharpe_ratio  # Negative because we minimize
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        # Bounds: no short selling
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_return = np.sum(expected_returns * optimal_weights)
            portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'optimization_success': True
            }
        else:
            return {
                'weights': np.array([1/n_assets] * n_assets),
                'expected_return': np.mean(expected_returns),
                'volatility': np.sqrt(np.mean(np.diag(covariance_matrix))),
                'sharpe_ratio': 0,
                'optimization_success': False,
                'error': result.message
            }
    
    def minimum_variance_optimization(self, covariance_matrix: pd.DataFrame) -> Dict:
        """
        Minimum Variance optimization.
        """
        n_assets = len(covariance_matrix)
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(covariance_matrix, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        # Bounds: no short selling
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_volatility = np.sqrt(result.fun)
            
            return {
                'weights': optimal_weights,
                'volatility': portfolio_volatility,
                'optimization_success': True
            }
        else:
            return {
                'weights': np.array([1/n_assets] * n_assets),
                'volatility': np.sqrt(np.mean(np.diag(covariance_matrix))),
                'optimization_success': False,
                'error': result.message
            }
    
    def hierarchical_risk_parity(self, returns: pd.DataFrame, linkage_method: str = 'single') -> Dict:
        """
        Hierarchical Risk Parity (HRP) optimization.
        """
        # Calculate correlation matrix
        correlation_matrix = returns.corr()
        
        # Convert correlation to distance matrix
        distance_matrix = np.sqrt(0.5 * (1 - correlation_matrix))
        
        # Perform hierarchical clustering
        from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
        
        # Create linkage matrix
        linkage_matrix = linkage(distance_matrix, method=linkage_method)
        
        # Get cluster assignments
        n_clusters = len(returns.columns)
        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Calculate weights using HRP algorithm
        weights = self._hrp_weights(returns, clusters)
        
        # Calculate portfolio metrics
        covariance_matrix = returns.cov()
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        
        return {
            'weights': weights,
            'portfolio_volatility': portfolio_volatility,
            'clusters': clusters,
            'linkage_matrix': linkage_matrix,
            'optimization_success': True
        }
    
    def _hrp_weights(self, returns: pd.DataFrame, clusters: np.ndarray) -> np.ndarray:
        """
        Calculate HRP weights based on cluster assignments.
        """
        n_assets = len(returns.columns)
        weights = np.zeros(n_assets)
        
        # Calculate variance for each asset
        variances = returns.var()
        
        # Calculate weights based on clusters
        unique_clusters = np.unique(clusters)
        
        for cluster in unique_clusters:
            cluster_assets = np.where(clusters == cluster)[0]
            cluster_variance = variances.iloc[cluster_assets].sum()
            
            # Distribute weight within cluster
            for asset_idx in cluster_assets:
                weights[asset_idx] = variances.iloc[asset_idx] / cluster_variance
        
        # Normalize weights
        weights = weights / weights.sum()
        
        return weights
    
    def black_litterman_optimization(self, returns: pd.Series, covariance_matrix: pd.DataFrame,
                                   market_caps: pd.Series, views: Dict = None) -> Dict:
        """
        Black-Litterman optimization with market equilibrium and views.
        """
        n_assets = len(returns)
        
        # Market equilibrium returns (reverse optimization)
        market_weights = market_caps / market_caps.sum()
        market_return = np.sum(returns.mean() * market_weights)
        market_volatility = np.sqrt(np.dot(market_weights.T, np.dot(covariance_matrix, market_weights)))
        
        # Risk aversion parameter
        risk_aversion = (market_return - self.risk_free_rate) / (market_volatility ** 2)
        
        # Equilibrium returns
        equilibrium_returns = risk_aversion * np.dot(covariance_matrix, market_weights)
        
        # If no views provided, use equilibrium returns
        if views is None:
            final_returns = equilibrium_returns
        else:
            # Process views (simplified implementation)
            # In practice, this would involve more complex view processing
            final_returns = equilibrium_returns
        
        # Optimize with final returns
        return self.mean_variance_optimization(
            pd.Series(final_returns, index=returns.index),
            covariance_matrix
        )
    
    def machine_learning_optimization(self, returns: pd.DataFrame, 
                                    method: str = 'clustering') -> Dict:
        """
        Machine learning-based portfolio optimization.
        """
        if method == 'clustering':
            return self._clustering_optimization(returns)
        elif method == 'pca':
            return self._pca_optimization(returns)
        else:
            raise ValueError(f"Unknown ML optimization method: {method}")
    
    def _clustering_optimization(self, returns: pd.DataFrame) -> Dict:
        """
        Portfolio optimization using K-means clustering.
        """
        # Calculate correlation matrix
        correlation_matrix = returns.corr()
        
        # Perform PCA for dimensionality reduction
        pca = PCA(n_components=min(5, len(returns.columns)))
        features = pca.fit_transform(correlation_matrix)
        
        # Perform clustering
        n_clusters = min(3, len(returns.columns))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # Calculate weights based on clusters
        weights = np.zeros(len(returns.columns))
        
        for cluster_id in range(n_clusters):
            cluster_assets = np.where(clusters == cluster_id)[0]
            cluster_weight = 1.0 / n_clusters  # Equal weight to each cluster
            
            # Distribute weight within cluster
            for asset_idx in cluster_assets:
                weights[asset_idx] = cluster_weight / len(cluster_assets)
        
        # Calculate portfolio metrics
        covariance_matrix = returns.cov()
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        
        return {
            'weights': weights,
            'portfolio_volatility': portfolio_volatility,
            'clusters': clusters,
            'method': 'clustering',
            'optimization_success': True
        }
    
    def _pca_optimization(self, returns: pd.DataFrame) -> Dict:
        """
        Portfolio optimization using Principal Component Analysis.
        """
        # Calculate correlation matrix
        correlation_matrix = returns.corr()
        
        # Perform PCA
        pca = PCA()
        pca.fit(correlation_matrix)
        
        # Use first few principal components
        n_components = min(3, len(returns.columns))
        principal_components = pca.components_[:n_components]
        
        # Calculate weights based on principal components
        weights = np.mean(principal_components, axis=0)
        weights = np.abs(weights)  # Take absolute values
        weights = weights / weights.sum()  # Normalize
        
        # Calculate portfolio metrics
        covariance_matrix = returns.cov()
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        
        return {
            'weights': weights,
            'portfolio_volatility': portfolio_volatility,
            'explained_variance': pca.explained_variance_ratio_[:n_components],
            'method': 'pca',
            'optimization_success': True
        }
    
    def compare_optimization_methods(self, returns: pd.DataFrame, 
                                   market_caps: Optional[pd.Series] = None) -> Dict:
        """
        Compare different optimization methods.
        """
        # Calculate returns and covariance
        returns_series = returns.mean()
        covariance_matrix = returns.cov()
        
        # Run different optimization methods
        results = {}
        
        # Mean-Variance Optimization
        results['mean_variance'] = self.mean_variance_optimization(
            returns_series, covariance_matrix
        )
        
        # Maximum Sharpe Ratio
        results['max_sharpe'] = self.maximum_sharpe_optimization(
            returns_series, covariance_matrix
        )
        
        # Minimum Variance
        results['min_variance'] = self.minimum_variance_optimization(covariance_matrix)
        
        # Risk Parity
        results['risk_parity'] = self.risk_parity_optimization(covariance_matrix)
        
        # Hierarchical Risk Parity
        results['hrp'] = self.hierarchical_risk_parity(returns)
        
        # Machine Learning (Clustering)
        results['ml_clustering'] = self.machine_learning_optimization(returns, 'clustering')
        
        # Machine Learning (PCA)
        results['ml_pca'] = self.machine_learning_optimization(returns, 'pca')
        
        # Black-Litterman (if market caps provided)
        if market_caps is not None:
            results['black_litterman'] = self.black_litterman_optimization(
                returns_series, covariance_matrix, market_caps
            )
        
        # Calculate comparison metrics
        comparison = {}
        for method, result in results.items():
            if result['optimization_success']:
                weights = result['weights']
                portfolio_return = np.sum(returns_series * weights)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
                
                comparison[method] = {
                    'weights': weights,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_weight': np.max(weights),
                    'min_weight': np.min(weights),
                    'weight_concentration': np.sum(weights ** 2)  # Herfindahl index
                }
        
        return {
            'methods': comparison,
            'best_sharpe': max(comparison.items(), key=lambda x: x[1]['sharpe_ratio']),
            'best_volatility': min(comparison.items(), key=lambda x: x[1]['volatility']),
            'best_return': max(comparison.items(), key=lambda x: x[1]['expected_return'])
        }
    
    def rebalance_portfolio(self, current_weights: np.ndarray, target_weights: np.ndarray,
                          rebalancing_threshold: float = 0.05) -> Dict:
        """
        Determine rebalancing trades based on current and target weights.
        """
        weight_diff = target_weights - current_weights
        
        # Find assets that need rebalancing
        rebalance_mask = np.abs(weight_diff) > rebalancing_threshold
        rebalance_assets = np.where(rebalance_mask)[0]
        
        # Calculate rebalancing trades
        trades = []
        for asset_idx in rebalance_assets:
            trade = {
                'asset_index': asset_idx,
                'current_weight': current_weights[asset_idx],
                'target_weight': target_weights[asset_idx],
                'weight_change': weight_diff[asset_idx],
                'trade_type': 'BUY' if weight_diff[asset_idx] > 0 else 'SELL'
            }
            trades.append(trade)
        
        # Calculate rebalancing cost (simplified)
        total_turnover = np.sum(np.abs(weight_diff))
        
        return {
            'trades': trades,
            'total_turnover': total_turnover,
            'rebalancing_cost': total_turnover * 0.001,  # Assume 0.1% transaction cost
            'assets_to_rebalance': len(rebalance_assets)
        } 