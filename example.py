import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import project modules
from data_utils import DataLoader
from regime_detection import MarketRegimeDetector
from signal_modeling import (
    MomentumSignal, VolatilityBreakoutSignal, 
    RSIDivergenceSignal, MASignal, SignalBacktester, SignalCombiner
)
from portfolio import PortfolioConstructor
from risk_management import RiskManager

def main():
    """
    Example script demonstrating AlphaTrace AI functionality.
    """
    print("AlphaTrace AI - Example Script")
    print("=" * 50)
    
    # 1. Data Loading
    print("\n1. Loading market data...")
    data_loader = DataLoader()
    
    # Define tickers and date range
    tickers = ["SPY", "QQQ", "GLD", "TLT", "IWM"]
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    
    # Load historical data
    data_dict = data_loader.load_historical_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        interval='1d'
    )
    
    # Preprocess data
    processed_data = data_loader.preprocess_data(data_dict)
    
    # Create market index
    market_index = data_loader.create_market_index(processed_data)
    
    # Use SPY for regime detection
    main_ticker = "SPY"
    main_data = processed_data[main_ticker]
    
    print(f"Loaded data for {len(tickers)} tickers from {start_date} to {end_date}")
    print(f"Sample data for {main_ticker}:")
    print(main_data.head())
    
    # 2. Regime Detection
    print("\n2. Detecting market regimes...")
    
    regime_detector = MarketRegimeDetector(
        method="kmeans",
        n_regimes=4,
        lookback_window=60,
        features=["returns", "volatility", "momentum", "trend"]
    )
    
    # Fit regime detector and make predictions
    regime_detector.fit(main_data)
    data_with_regimes = regime_detector.predict_regimes(main_data)
    
    # Extract regime series
    regimes = data_with_regimes['regime']
    
    # Get regime statistics
    regime_stats = regime_detector.get_regime_statistics(data_with_regimes)
    
    print("\nRegime statistics:")
    print(regime_stats)
    
    # Plot regimes
    regime_fig = regime_detector.visualize_regimes(data_with_regimes)
    regime_fig.savefig('regimes.png')
    print("Saved regime visualization to 'regimes.png'")
    
    # 3. Signal Generation & Backtesting
    print("\n3. Generating and backtesting signals...")
    
    # Create signals
    signals = [
        MomentumSignal(lookback_period=20),
        MomentumSignal(lookback_period=60),
        VolatilityBreakoutSignal(window=20, num_std=2.0),
        RSIDivergenceSignal(rsi_period=14, divergence_window=10),
        MASignal(fast_period=50, slow_period=200)
    ]
    
    print(f"Created {len(signals)} signals:")
    for signal in signals:
        print(f"  - {signal.name}")
    
    # Create backtester
    backtester = SignalBacktester(
        signals=signals,
        forward_returns_period=5,
        transaction_cost=0.001
    )
    
    # Backtest signals by regime
    signal_performance_by_regime = backtester.backtest_signals_by_regime(data_with_regimes)
    
    # Generate signal features
    signal_features = backtester.generate_signal_features(main_data)
    
    print("\nSignal performance for Regime 0 (Bull):")
    print(signal_performance_by_regime[0].head())
    
    # 4. Portfolio Construction
    print("\n4. Constructing adaptive portfolio...")
    
    # Calculate signal returns (assuming 1-day forward returns for simplicity)
    signal_returns = pd.DataFrame(index=signal_features.index[1:])
    
    for signal_name in signal_features.columns:
        # Use signal direction to calculate returns
        signal = signal_features[signal_name].dropna()
        if len(signal) > 0:
            # Standardize signal
            signal = (signal - signal.mean()) / signal.std()
            
            # Calculate returns based on signal direction
            signal_returns[signal_name] = np.sign(signal.shift(1)) * main_data['Returns'].iloc[1:]
    
    # Initialize portfolio constructor
    portfolio_constructor = PortfolioConstructor(
        min_weight=0.0,
        max_weight=0.3,
        transaction_cost=0.001
    )
    
    # Backtest portfolio
    portfolio_results = portfolio_constructor.backtest_portfolio(
        signal_returns=signal_returns,
        regimes=regimes.iloc[1:] if len(regimes) > 1 else None,
        rebalance_frequency=5,
        top_n_signals=3
    )
    
    print("\nPortfolio backtest results summary:")
    print(f"Final portfolio value: {portfolio_results['portfolio_value'].iloc[-1]:.2f}")
    print(f"Total return: {portfolio_results['portfolio_value'].iloc[-1] / portfolio_results['portfolio_value'].iloc[0] - 1:.2%}")
    print(f"Maximum drawdown: {portfolio_results['drawdown'].min():.2%}")
    
    # Plot portfolio results
    portfolio_fig = portfolio_constructor.plot_backtest_results(portfolio_results)
    portfolio_fig.savefig('portfolio.png')
    print("Saved portfolio visualization to 'portfolio.png'")
    
    # 5. Risk Analysis
    print("\n5. Risk analysis and performance reporting...")
    
    risk_manager = RiskManager()
    
    # Extract portfolio returns
    portfolio_returns = portfolio_results['portfolio_return']
    
    # Calculate performance metrics
    metrics = risk_manager.calculate_metrics(portfolio_returns)
    
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        if 'return' in metric or 'drawdown' in metric:
            print(f"  - {metric}: {value:.2%}")
        else:
            print(f"  - {metric}: {value:.2f}")
    
    # Calculate alpha and beta
    alpha, beta = risk_manager.calculate_alpha_beta(
        portfolio_returns, 
        main_data['Returns'].iloc[1:]
    )
    
    print(f"\nAlpha: {alpha:.2%}")
    print(f"Beta: {beta:.2f}")
    
    # Perform stress tests
    stress_tests = risk_manager.stress_test(portfolio_returns)
    
    print("\nStress test results:")
    print(stress_tests)
    
    # Plot drawdowns
    drawdown_fig = risk_manager.plot_drawdowns(portfolio_returns)
    drawdown_fig.savefig('drawdowns.png')
    print("Saved drawdowns visualization to 'drawdowns.png'")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 