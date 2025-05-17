import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# Import project modules
from data_utils import DataLoader
from regime_detection import MarketRegimeDetector
from signal_modeling import (
    MomentumSignal, VolatilityBreakoutSignal, 
    RSIDivergenceSignal, MASignal, SignalBacktester
)
from portfolio import PortfolioConstructor
from risk_management import RiskManager

def run_test():
    """
    Run a simple test to validate AlphaTrace AI implementation.
    """
    print("AlphaTrace AI - Test Suite")
    print("=" * 50)
    
    # Test data loading
    print("\nTesting DataLoader...")
    try:
        data_loader = DataLoader()
        
        # Use a small sample for testing
        tickers = ["SPY"]
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        data_dict = data_loader.load_historical_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date
        )
        
        processed_data = data_loader.preprocess_data(data_dict)
        
        if len(processed_data) > 0 and "SPY" in processed_data:
            print("✓ DataLoader test successful")
        else:
            print("✗ DataLoader test failed")
    except Exception as e:
        print(f"✗ DataLoader test failed with error: {str(e)}")
    
    # Test regime detection
    print("\nTesting MarketRegimeDetector...")
    try:
        main_data = processed_data["SPY"]
        
        regime_detector = MarketRegimeDetector(
            method="kmeans",
            n_regimes=4,
            lookback_window=20
        )
        
        regime_detector.fit(main_data)
        data_with_regimes = regime_detector.predict_regimes(main_data)
        
        if "regime" in data_with_regimes.columns:
            unique_regimes = data_with_regimes["regime"].nunique()
            print(f"✓ MarketRegimeDetector test successful (detected {unique_regimes} regimes)")
        else:
            print("✗ MarketRegimeDetector test failed")
    except Exception as e:
        print(f"✗ MarketRegimeDetector test failed with error: {str(e)}")
    
    # Test signal generation
    print("\nTesting Signal Generation...")
    try:
        signals = [
            MomentumSignal(lookback_period=20),
            VolatilityBreakoutSignal(window=20, num_std=2.0),
            RSIDivergenceSignal(rsi_period=14, divergence_window=10),
            MASignal(fast_period=50, slow_period=200)
        ]
        
        successful_signals = []
        
        for signal in signals:
            try:
                signal_values = signal.generate(main_data)
                if not signal_values.empty:
                    successful_signals.append(signal.name)
            except:
                pass
        
        if len(successful_signals) > 0:
            print(f"✓ Signal generation test successful: {', '.join(successful_signals)}")
        else:
            print("✗ Signal generation test failed")
    except Exception as e:
        print(f"✗ Signal generation test failed with error: {str(e)}")
    
    # Test backtesting
    print("\nTesting SignalBacktester...")
    try:
        backtester = SignalBacktester(
            signals=signals,
            forward_returns_period=5,
            transaction_cost=0.001
        )
        
        # Backtest just one signal for speed
        result = backtester.backtest_signal(signals[0], main_data)
        
        if result and "sharpe" in result:
            print(f"✓ SignalBacktester test successful (Sharpe ratio: {result['sharpe']:.2f})")
        else:
            print("✗ SignalBacktester test failed")
    except Exception as e:
        print(f"✗ SignalBacktester test failed with error: {str(e)}")
    
    # Test portfolio construction
    print("\nTesting PortfolioConstructor...")
    try:
        # Generate signal features
        signal_features = backtester.generate_signal_features(main_data)
        
        # Calculate signal returns
        signal_returns = pd.DataFrame(index=signal_features.index[1:])
        
        for signal_name in signal_features.columns:
            signal = signal_features[signal_name].dropna()
            if len(signal) > 0:
                signal = (signal - signal.mean()) / signal.std()
                signal_returns[signal_name] = np.sign(signal.shift(1)) * main_data['Returns'].iloc[1:]
        
        # Initialize portfolio constructor
        portfolio_constructor = PortfolioConstructor(
            min_weight=0.0,
            max_weight=0.3,
            transaction_cost=0.001
        )
        
        # Calculate optimal weights (just testing function, not full backtest)
        weights = portfolio_constructor.calculate_optimal_weights(
            signal_returns.iloc[:30],  # Small sample for quick test
            top_n_signals=2
        )
        
        if len(weights) > 0:
            print(f"✓ PortfolioConstructor test successful")
        else:
            print("✗ PortfolioConstructor test failed")
    except Exception as e:
        print(f"✗ PortfolioConstructor test failed with error: {str(e)}")
    
    # Test risk management
    print("\nTesting RiskManager...")
    try:
        risk_manager = RiskManager()
        
        # Create some dummy returns
        returns = pd.Series(
            np.random.normal(0.0005, 0.01, 252),
            index=pd.date_range(end=datetime.now(), periods=252, freq='B')
        )
        
        # Calculate metrics
        metrics = risk_manager.calculate_metrics(returns)
        
        if metrics and "sharpe_ratio" in metrics:
            print(f"✓ RiskManager test successful (Sharpe ratio: {metrics['sharpe_ratio']:.2f})")
        else:
            print("✗ RiskManager test failed")
    except Exception as e:
        print(f"✗ RiskManager test failed with error: {str(e)}")
    
    print("\nTest suite complete.")

if __name__ == "__main__":
    run_test() 