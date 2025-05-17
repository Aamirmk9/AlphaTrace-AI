import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, timedelta

class RiskManager:
    """
    Risk management and performance analysis module.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the risk manager.
        
        Args:
            risk_free_rate: Annual risk-free rate
        """
        self.risk_free_rate = risk_free_rate / 252  # Daily risk-free rate
        
    def calculate_metrics(self, returns: pd.Series) -> Dict:
        """
        Calculate performance metrics for a return series.
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary with performance metrics
        """
        # Filter out NaN values
        returns = returns.dropna()
        
        if len(returns) == 0:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0
            }
        
        # Calculate total return
        total_return = (1 + returns).cumprod().iloc[-1] - 1
        
        # Calculate annualized return
        n_days = len(returns)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1
        
        # Calculate volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio
        sharpe_ratio = (annualized_return - self.risk_free_rate * 252) / volatility if volatility > 0 else 0
        
        # Calculate Sortino ratio (downside risk only)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - self.risk_free_rate * 252) / downside_deviation if downside_deviation > 0 else 0
        
        # Calculate maximum drawdown
        cum_returns = (1 + returns).cumprod()
        max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()
        
        # Calculate win rate
        win_rate = (returns > 0).mean()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }

    def calculate_alpha_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> Tuple[float, float]:
        """
        Calculate alpha and beta relative to a benchmark.
        
        Args:
            returns: Series of portfolio returns
            benchmark_returns: Series of benchmark returns
            
        Returns:
            Tuple of (alpha, beta)
        """
        # Align return series
        common_index = returns.index.intersection(benchmark_returns.index)
        strategy_returns = returns.loc[common_index]
        market_returns = benchmark_returns.loc[common_index]
        
        # Calculate covariance
        covariance = strategy_returns.cov(market_returns)
        market_variance = market_returns.var()
        
        # Calculate beta
        beta = covariance / market_variance if market_variance > 0 else 0
        
        # Calculate alpha (annualized)
        alpha = (strategy_returns.mean() - beta * market_returns.mean()) * 252
        
        return alpha, beta
    
    def calculate_drawdowns(self, returns: pd.Series) -> pd.DataFrame:
        """
        Calculate drawdowns for a return series.
        
        Args:
            returns: Series of returns
            
        Returns:
            DataFrame with drawdown information
        """
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cum_returns.cummax()
        
        # Calculate drawdowns
        drawdowns = (cum_returns / running_max - 1)
        
        # Find drawdown periods
        is_in_drawdown = drawdowns < 0
        
        # Find start and end of drawdown periods
        drawdown_starts = is_in_drawdown & ~is_in_drawdown.shift(1).fillna(False)
        drawdown_ends = (~is_in_drawdown & is_in_drawdown.shift(1).fillna(False)) | (is_in_drawdown & (drawdowns == drawdowns.cummin()))
        
        # Create DataFrame for major drawdowns
        drawdown_periods = []
        
        current_drawdown_start = None
        
        for i, (date, is_start, is_end, drawdown) in enumerate(zip(
            returns.index, drawdown_starts, drawdown_ends, drawdowns)):
            
            if is_start:
                current_drawdown_start = date
                
            if is_end and current_drawdown_start is not None:
                # Find the maximum drawdown in this period
                period_mask = (returns.index >= current_drawdown_start) & (returns.index <= date)
                max_drawdown = drawdowns[period_mask].min()
                
                # Only include significant drawdowns (e.g., > 5%)
                if max_drawdown < -0.05:
                    start_value = cum_returns[current_drawdown_start]
                    end_value = cum_returns[date]
                    recovery_pct = (end_value / start_value) - 1
                    
                    drawdown_periods.append({
                        'start_date': current_drawdown_start,
                        'end_date': date,
                        'duration': (date - current_drawdown_start).days,
                        'max_drawdown': max_drawdown,
                        'recovery_pct': recovery_pct
                    })
                    
                current_drawdown_start = None
                
        return pd.DataFrame(drawdown_periods)

    def stress_test(self, portfolio_returns: pd.Series, 
                   scenarios: Dict[str, Dict[str, float]] = None) -> pd.DataFrame:
        """
        Perform stress tests on the portfolio.
        
        Args:
            portfolio_returns: Series of portfolio returns
            scenarios: Dictionary mapping scenario names to return modifications
            
        Returns:
            DataFrame with stress test results
        """
        # Default scenarios if none provided
        if scenarios is None:
            scenarios = {
                'market_crash': {'mean': -0.03, 'volatility': 2.0, 'duration': 20},
                'recession': {'mean': -0.01, 'volatility': 1.5, 'duration': 60},
                'volatility_spike': {'mean': 0, 'volatility': 3.0, 'duration': 10},
                'flat_market': {'mean': 0, 'volatility': 0.5, 'duration': 30}
            }
            
        results = []
        
        # Calculate baseline metrics
        baseline_metrics = self.calculate_metrics(portfolio_returns)
        baseline_equity = (1 + portfolio_returns).cumprod().iloc[-1]
        
        for scenario_name, params in scenarios.items():
            # Get parameters
            mean_shift = params.get('mean', 0)
            vol_multiplier = params.get('volatility', 1.0)
            duration = params.get('duration', 20)
            
            # Generate scenario returns using bootstrap and modification
            scenario_returns = portfolio_returns.copy()
            
            # Modify the last 'duration' days
            if len(scenario_returns) > duration:
                scenario_returns.iloc[-duration:] = (scenario_returns.iloc[-duration:] + mean_shift) * vol_multiplier
                
            # Calculate scenario metrics
            scenario_metrics = self.calculate_metrics(scenario_returns)
            scenario_equity = (1 + scenario_returns).cumprod().iloc[-1]
            
            # Calculate impact
            impact = {
                'scenario': scenario_name,
                'equity_impact_pct': (scenario_equity / baseline_equity - 1) * 100,
                'return_impact': scenario_metrics['annualized_return'] - baseline_metrics['annualized_return'],
                'volatility_impact': scenario_metrics['volatility'] - baseline_metrics['volatility'],
                'sharpe_impact': scenario_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio'],
                'max_drawdown_impact': scenario_metrics['max_drawdown'] - baseline_metrics['max_drawdown']
            }
            
            results.append(impact)
            
        return pd.DataFrame(results)
    
    def plot_returns(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None, 
                    figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot cumulative returns.
        
        Args:
            returns: Series of portfolio returns
            benchmark_returns: Series of benchmark returns (optional)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate cumulative returns
        portfolio_cum_returns = (1 + returns).cumprod()
        ax.plot(portfolio_cum_returns.index, portfolio_cum_returns, label='Portfolio')
        
        if benchmark_returns is not None:
            # Align benchmark returns to portfolio returns
            common_index = returns.index.intersection(benchmark_returns.index)
            benchmark_cum_returns = (1 + benchmark_returns.loc[common_index]).cumprod()
            ax.plot(benchmark_cum_returns.index, benchmark_cum_returns, label='Benchmark', linestyle='--')
            
        ax.set_title('Cumulative Returns')
        ax.set_ylabel('Growth of $1')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_drawdowns(self, returns: pd.Series, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot drawdowns.
        
        Args:
            returns: Series of portfolio returns
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate cumulative returns and drawdowns
        cum_returns = (1 + returns).cumprod()
        drawdowns = (cum_returns / cum_returns.cummax() - 1) * 100  # Convert to percentage
        
        ax.fill_between(drawdowns.index, drawdowns, 0, where=drawdowns < 0, color='red', alpha=0.3)
        ax.plot(drawdowns.index, drawdowns, color='red', linestyle='-', linewidth=1)
        
        ax.set_title('Portfolio Drawdowns')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True)
        
        return fig
    
    def plot_rolling_metrics(self, returns: pd.Series, window: int = 60, 
                            figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Plot rolling performance metrics.
        
        Args:
            returns: Series of portfolio returns
            window: Rolling window size in days
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Calculate rolling metrics
        rolling_return = returns.rolling(window).mean() * 252
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_return / rolling_vol
        
        # Plot rolling annualized return
        axes[0].plot(rolling_return.index, rolling_return * 100)  # Convert to percentage
        axes[0].set_title(f'Rolling {window}-Day Annualized Return')
        axes[0].set_ylabel('Return (%)')
        axes[0].grid(True)
        
        # Plot rolling annualized volatility
        axes[1].plot(rolling_vol.index, rolling_vol * 100)  # Convert to percentage
        axes[1].set_title(f'Rolling {window}-Day Annualized Volatility')
        axes[1].set_ylabel('Volatility (%)')
        axes[1].grid(True)
        
        # Plot rolling Sharpe ratio
        axes[2].plot(rolling_sharpe.index, rolling_sharpe)
        axes[2].set_title(f'Rolling {window}-Day Sharpe Ratio')
        axes[2].set_ylabel('Sharpe Ratio')
        axes[2].grid(True)
        
        plt.tight_layout()
        
        return fig
    
    def generate_report(self, portfolio_returns: pd.Series, 
                       benchmark_returns: Optional[pd.Series] = None,
                       regimes: Optional[pd.Series] = None) -> Dict:
        """
        Generate a comprehensive performance report.
        
        Args:
            portfolio_returns: Series of portfolio returns
            benchmark_returns: Series of benchmark returns (optional)
            regimes: Series of market regime labels (optional)
            
        Returns:
            Dictionary with report data
        """
        # Calculate overall metrics
        metrics = self.calculate_metrics(portfolio_returns)
        
        # Calculate alpha and beta if benchmark is provided
        if benchmark_returns is not None:
            alpha, beta = self.calculate_alpha_beta(portfolio_returns, benchmark_returns)
            metrics['alpha'] = alpha
            metrics['beta'] = beta
        
        # Perform stress tests
        stress_test_results = self.stress_test(portfolio_returns)
        
        # Calculate regime-specific performance if regimes are provided
        regime_performance = None
        if regimes is not None:
            regime_performance = {}
            
            # Align regimes with returns
            common_index = portfolio_returns.index.intersection(regimes.index)
            aligned_returns = portfolio_returns.loc[common_index]
            aligned_regimes = regimes.loc[common_index]
            
            # Calculate metrics for each regime
            for regime in aligned_regimes.unique():
                regime_returns = aligned_returns[aligned_regimes == regime]
                if len(regime_returns) > 0:
                    regime_performance[int(regime)] = self.calculate_metrics(regime_returns)
        
        # Create report dictionary
        report = {
            'metrics': metrics,
            'stress_tests': stress_test_results,
            'regime_performance': regime_performance
        }
        
        return report
    
    def plot_interactive_report(self, portfolio_returns: pd.Series, 
                               benchmark_returns: Optional[pd.Series] = None,
                               regimes: Optional[pd.Series] = None) -> Dict:
        """
        Generate interactive Plotly charts for a performance report.
        
        Args:
            portfolio_returns: Series of portfolio returns
            benchmark_returns: Series of benchmark returns (optional)
            regimes: Series of market regime labels (optional)
            
        Returns:
            Dictionary with Plotly figures
        """
        figures = {}
        
        # Cumulative returns chart
        cum_returns = (1 + portfolio_returns).cumprod()
        
        fig_cum_returns = go.Figure()
        fig_cum_returns.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns, 
                                           mode='lines', name='Portfolio'))
        
        if benchmark_returns is not None:
            # Align benchmark returns
            common_index = portfolio_returns.index.intersection(benchmark_returns.index)
            aligned_benchmark = benchmark_returns.loc[common_index]
            benchmark_cum_returns = (1 + aligned_benchmark).cumprod()
            
            fig_cum_returns.add_trace(go.Scatter(x=benchmark_cum_returns.index, 
                                               y=benchmark_cum_returns, 
                                               mode='lines', name='Benchmark'))
            
        fig_cum_returns.update_layout(title='Cumulative Returns',
                                    xaxis_title='Date',
                                    yaxis_title='Growth of $1',
                                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        
        figures['cumulative_returns'] = fig_cum_returns
        
        # Drawdowns chart
        drawdowns = (cum_returns / cum_returns.cummax() - 1) * 100  # Convert to percentage
        
        fig_drawdowns = go.Figure()
        fig_drawdowns.add_trace(go.Scatter(x=drawdowns.index, y=drawdowns, 
                                         mode='lines', name='Drawdown',
                                         fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.3)'))
        
        fig_drawdowns.update_layout(title='Portfolio Drawdowns',
                                  xaxis_title='Date',
                                  yaxis_title='Drawdown (%)')
        
        figures['drawdowns'] = fig_drawdowns
        
        # Regime-specific performance if regimes are provided
        if regimes is not None:
            # Align regimes with returns
            common_index = portfolio_returns.index.intersection(regimes.index)
            aligned_returns = portfolio_returns.loc[common_index]
            aligned_regimes = regimes.loc[common_index]
            
            # Convert to categorical if not already
            if not pd.api.types.is_categorical_dtype(aligned_regimes):
                aligned_regimes = aligned_regimes.astype('category')
                
            # Create cumulative returns by regime
            fig_regime_returns = go.Figure()
            
            for regime in aligned_regimes.cat.categories:
                regime_mask = aligned_regimes == regime
                if regime_mask.sum() > 0:
                    regime_returns = (1 + aligned_returns[regime_mask]).cumprod()
                    
                    fig_regime_returns.add_trace(go.Scatter(
                        x=regime_returns.index, 
                        y=regime_returns, 
                        mode='lines', 
                        name=f'Regime {regime}'
                    ))
                    
            fig_regime_returns.update_layout(title='Returns by Market Regime',
                                          xaxis_title='Date',
                                          yaxis_title='Growth of $1')
                                          
            figures['regime_returns'] = fig_regime_returns
            
        return figures 