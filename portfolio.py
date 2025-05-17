import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class PortfolioConstructor:
    """
    Constructs adaptive portfolios based on market regimes and signal performance.
    """
    
    def __init__(self, 
                 min_weight: float = 0.0, 
                 max_weight: float = 0.3,
                 risk_free_rate: float = 0.02,
                 transaction_cost: float = 0.001,
                 regularization_lambda: float = 0.1):
        """
        Initialize the portfolio constructor.
        
        Args:
            min_weight: Minimum weight for any asset
            max_weight: Maximum weight for any asset
            risk_free_rate: Annual risk-free rate
            transaction_cost: Transaction cost per trade
            regularization_lambda: L2 regularization parameter
        """
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.risk_free_rate = risk_free_rate / 252  # Daily risk-free rate
        self.transaction_cost = transaction_cost
        self.regularization_lambda = regularization_lambda
        self.current_weights = None
        self.weight_history = []
        
    def mean_variance_optimization(self, 
                                   returns: pd.DataFrame, 
                                   risk_aversion: float = 1.0,
                                   previous_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform mean-variance optimization with L2 regularization.
        
        Args:
            returns: DataFrame of asset returns
            risk_aversion: Risk aversion parameter
            previous_weights: Previous portfolio weights (for regularization)
            
        Returns:
            Array of optimal weights
        """
        n_assets = returns.shape[1]
        
        # Prepare expected returns and covariance matrix
        expected_returns = returns.mean().values
        cov_matrix = returns.cov().values
        
        # Set initial weights
        initial_weights = np.ones(n_assets) / n_assets
        
        # If no previous weights, use equal weighting for regularization
        if previous_weights is None:
            previous_weights = initial_weights
            
        # Define objective function (negative Sharpe ratio with regularization)
        def objective(weights):
            # Portfolio return and variance
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            
            # L2 regularization term
            l2_penalty = self.regularization_lambda * np.sum((weights - previous_weights) ** 2)
            
            # Mean-variance utility with risk aversion
            utility = portfolio_return - (0.5 * risk_aversion * portfolio_variance) - l2_penalty
            
            # Return negative utility for minimization
            return -utility
        
        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        
        # Bounds for each weight
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        
        # Perform optimization
        result = minimize(
            objective, 
            initial_weights, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        # Return optimal weights
        if result.success:
            return result.x
        else:
            # Fallback to initial weights if optimization fails
            return initial_weights
        
    def calculate_optimal_weights(self, 
                                  signals_data: pd.DataFrame, 
                                  regime: Optional[int] = None,
                                  top_n_signals: int = 5,
                                  min_sharpe: float = 0.5) -> np.ndarray:
        """
        Calculate optimal weights for top-performing signals in a given regime.
        
        Args:
            signals_data: DataFrame with signal return data
            regime: Current market regime
            top_n_signals: Number of top signals to include
            min_sharpe: Minimum Sharpe ratio for signal inclusion
            
        Returns:
            Array of optimal weights
        """
        # Filter signals by Sharpe ratio
        valid_signals = signals_data.columns[
            signals_data.mean() / signals_data.std() * np.sqrt(252) > min_sharpe
        ]
        
        if len(valid_signals) == 0:
            # If no valid signals, return cash position (all zeros)
            return np.zeros(len(signals_data.columns))
        
        # Calculate Sharpe ratios
        sharpes = (signals_data[valid_signals].mean() / 
                   signals_data[valid_signals].std() * np.sqrt(252))
        
        # Select top N signals
        top_signals = sharpes.sort_values(ascending=False).head(top_n_signals).index
        
        if len(top_signals) == 0:
            # If no top signals, return cash position
            return np.zeros(len(signals_data.columns))
            
        # Optimize weights for top signals
        signal_returns = signals_data[top_signals]
        optimal_weights_subset = self.mean_variance_optimization(
            signal_returns, 
            previous_weights=self.current_weights
        )
        
        # Map back to full signal space
        full_weights = np.zeros(len(signals_data.columns))
        for i, signal in enumerate(top_signals):
            idx = signals_data.columns.get_loc(signal)
            full_weights[idx] = optimal_weights_subset[i]
            
        return full_weights
    
    def rebalance_portfolio(self, 
                           signal_returns: pd.DataFrame, 
                           current_regime: Optional[int] = None,
                           top_n_signals: int = 5,
                           lookback_window: int = 60) -> pd.Series:
        """
        Rebalance portfolio based on recent signal performance.
        
        Args:
            signal_returns: DataFrame with signal returns time series
            current_regime: Current market regime (optional)
            top_n_signals: Number of top signals to include
            lookback_window: Window for calculating recent performance
            
        Returns:
            Series with new weights
        """
        # Use recent returns for optimization
        recent_returns = signal_returns.iloc[-lookback_window:]
        
        # Calculate optimal weights
        optimal_weights = self.calculate_optimal_weights(
            recent_returns, 
            regime=current_regime,
            top_n_signals=top_n_signals
        )
        
        # Update current weights
        self.current_weights = optimal_weights
        
        # Create weight Series
        weights = pd.Series(optimal_weights, index=signal_returns.columns)
        
        # Store in history
        self.weight_history.append({
            'date': signal_returns.index[-1],
            'weights': weights,
            'regime': current_regime
        })
        
        return weights
    
    def get_transaction_costs(self, old_weights: pd.Series, new_weights: pd.Series) -> float:
        """
        Calculate transaction costs for portfolio rebalancing.
        
        Args:
            old_weights: Previous portfolio weights
            new_weights: New portfolio weights
            
        Returns:
            Transaction cost
        """
        if old_weights is None:
            old_weights = pd.Series(0, index=new_weights.index)
            
        # Calculate absolute weight changes
        weight_changes = np.abs(new_weights - old_weights).sum()
        
        # Multiply by transaction cost
        return weight_changes * self.transaction_cost
    
    def backtest_portfolio(self, 
                           signal_returns: pd.DataFrame, 
                           regimes: Optional[pd.Series] = None,
                           rebalance_frequency: int = 5,
                           top_n_signals: int = 5,
                           lookback_window: int = 60) -> pd.DataFrame:
        """
        Backtest portfolio strategy.
        
        Args:
            signal_returns: DataFrame with signal returns
            regimes: Series with regime labels
            rebalance_frequency: Number of days between rebalancing
            top_n_signals: Number of top signals in portfolio
            lookback_window: Window for performance calculation
            
        Returns:
            DataFrame with backtest results
        """
        # Initialize backtest data
        backtest_data = pd.DataFrame(index=signal_returns.index)
        backtest_data['portfolio_return'] = 0.0
        
        # Current portfolio weights
        current_weights = None
        
        # Track portfolio value
        portfolio_value = 1.0
        backtest_data['portfolio_value'] = portfolio_value
        
        # Last rebalance date
        last_rebalance = signal_returns.index[0]
        
        # Loop through trading days
        for i in range(lookback_window, len(signal_returns)):
            current_date = signal_returns.index[i]
            
            # Determine if rebalancing is needed
            days_since_rebalance = (current_date - last_rebalance).days
            is_regime_change = False
            
            if regimes is not None:
                # Check if regime has changed
                if i > 0 and regimes.iloc[i] != regimes.iloc[i-1]:
                    is_regime_change = True
            
            # Rebalance if it's time or if regime has changed
            if days_since_rebalance >= rebalance_frequency or is_regime_change:
                # Get historical data for optimization
                historical_returns = signal_returns.iloc[i-lookback_window:i]
                
                # Get current regime
                current_regime = regimes.iloc[i] if regimes is not None else None
                
                # Calculate new weights
                new_weights = self.calculate_optimal_weights(
                    historical_returns,
                    regime=current_regime,
                    top_n_signals=top_n_signals
                )
                
                new_weights = pd.Series(new_weights, index=signal_returns.columns)
                
                # Calculate transaction costs
                txn_cost = self.get_transaction_costs(current_weights, new_weights)
                
                # Update current weights
                current_weights = new_weights
                
                # Store rebalance details
                last_rebalance = current_date
                
                # Apply transaction costs
                portfolio_value *= (1 - txn_cost)
            
            # Calculate today's portfolio return
            if current_weights is not None:
                daily_return = (signal_returns.iloc[i] * current_weights).sum()
            else:
                daily_return = 0.0
                
            # Update portfolio value
            portfolio_value *= (1 + daily_return)
            
            # Store results
            backtest_data.loc[current_date, 'portfolio_return'] = daily_return
            backtest_data.loc[current_date, 'portfolio_value'] = portfolio_value
            
            if regimes is not None:
                backtest_data.loc[current_date, 'regime'] = regimes.iloc[i]
                
        # Calculate cumulative returns
        backtest_data['cumulative_return'] = (1 + backtest_data['portfolio_return']).cumprod() - 1
        
        # Calculate drawdowns
        backtest_data['peak_value'] = backtest_data['portfolio_value'].cummax()
        backtest_data['drawdown'] = (backtest_data['portfolio_value'] / backtest_data['peak_value']) - 1
        
        return backtest_data
    
    def calculate_performance_metrics(self, backtest_data: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics from backtest results.
        
        Args:
            backtest_data: DataFrame with backtest results
            
        Returns:
            Dictionary with performance metrics
        """
        # Extract returns
        returns = backtest_data['portfolio_return']
        
        # Calculate metrics
        total_return = backtest_data['portfolio_value'].iloc[-1] / backtest_data['portfolio_value'].iloc[0] - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate * 252) / volatility if volatility > 0 else 0
        max_drawdown = backtest_data['drawdown'].min()
        
        # Calculate Sortino ratio (downside risk only)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - self.risk_free_rate * 252) / downside_deviation if downside_deviation > 0 else 0
        
        # Calculate Calmar ratio (return / max drawdown)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Calculate win rate
        win_rate = (returns > 0).mean()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
    
    def plot_backtest_results(self, backtest_data: pd.DataFrame, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot backtest results.
        
        Args:
            backtest_data: DataFrame with backtest results
            figsize: Figure size
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Plot portfolio value
        axes[0].plot(backtest_data.index, backtest_data['portfolio_value'], 'b-')
        axes[0].set_title('Portfolio Value')
        axes[0].grid(True)
        
        # Plot drawdowns
        axes[1].fill_between(
            backtest_data.index, 
            backtest_data['drawdown'] * 100, 
            0, 
            where=backtest_data['drawdown'] < 0, 
            color='r', 
            alpha=0.3
        )
        axes[1].set_title('Drawdown (%)')
        axes[1].grid(True)
        
        # Plot regime if available
        if 'regime' in backtest_data.columns:
            axes[2].scatter(
                backtest_data.index, 
                backtest_data['regime'], 
                c=backtest_data['regime'], 
                cmap='viridis', 
                s=10
            )
            axes[2].set_title('Market Regime')
            axes[2].grid(True)
            
        plt.tight_layout()
        
        return fig
    
    def plot_weight_evolution(self, signal_names: List[str], figsize: Tuple[int, int] = (12, 8)):
        """
        Plot the evolution of weights over time.
        
        Args:
            signal_names: List of signal names to plot
            figsize: Figure size
        """
        if not self.weight_history:
            raise ValueError("No weight history available. Run backtest first.")
            
        # Extract weight history
        dates = [entry['date'] for entry in self.weight_history]
        weights_df = pd.DataFrame(
            [entry['weights'] for entry in self.weight_history],
            index=dates
        )
        
        # Plot weights
        plt.figure(figsize=figsize)
        plt.stackplot(
            weights_df.index,
            [weights_df[col] for col in weights_df.columns],
            labels=weights_df.columns,
            alpha=0.8
        )
        
        plt.title('Portfolio Weight Evolution')
        plt.xlabel('Date')
        plt.ylabel('Weight')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)
        
        return plt.gcf()
    
    def stress_test(self, 
                    signal_returns: pd.DataFrame, 
                    scenarios: Dict[str, Dict[str, float]] = None) -> pd.DataFrame:
        """
        Stress test the portfolio under different scenarios.
        
        Args:
            signal_returns: DataFrame with signal returns
            scenarios: Dictionary mapping scenario names to shock factors
            
        Returns:
            DataFrame with stress test results
        """
        if self.current_weights is None:
            raise ValueError("No current weights available. Run backtest first.")
            
        # Default scenarios if none provided
        if scenarios is None:
            scenarios = {
                'market_crash': {'momentum': -0.5, 'volatility': 2.0},
                'bull_market': {'momentum': 1.5, 'value': 0.8},
                'flat_market': {'momentum': -0.2, 'mean_reversion': 1.2}
            }
            
        results = []
        
        # Run each stress scenario
        for scenario_name, shocks in scenarios.items():
            # Copy returns
            scenario_returns = signal_returns.copy()
            
            # Apply shocks to relevant signals
            for signal_type, factor in shocks.items():
                # Find signals containing the signal type in their name
                matching_cols = [col for col in scenario_returns.columns if signal_type.lower() in col.lower()]
                
                for col in matching_cols:
                    scenario_returns[col] *= factor
                    
            # Calculate portfolio return under this scenario
            scenario_return = (scenario_returns.mean() * self.current_weights).sum()
            
            # Store result
            results.append({
                'scenario': scenario_name,
                'return': scenario_return,
                'change': scenario_return - (signal_returns.mean() * self.current_weights).sum()
            })
            
        return pd.DataFrame(results) 