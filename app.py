import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
from typing import List, Dict, Tuple, Optional, Union

# Import project modules
from data_utils import DataLoader
from regime_detection import MarketRegimeDetector
from signal_modeling import (
    AlphaSignal, MomentumSignal, VolatilityBreakoutSignal, 
    RSIDivergenceSignal, MASignal, SignalBacktester, SignalCombiner
)
from portfolio import PortfolioConstructor
from risk_management import RiskManager

# Set page configuration
st.set_page_config(
    page_title="AlphaTrace AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("AlphaTrace AI")
st.markdown(
    """
    A real-time alpha generation and backtesting platform that dynamically adjusts 
    trading signals and portfolio weights based on market regimes.
    """
)

# Sidebar for inputs
st.sidebar.header("Data Settings")

# Ticker selection
tickers = st.sidebar.text_input(
    "Enter Ticker Symbols (comma-separated)",
    value="SPY,QQQ,GLD,TLT,IWM"
).split(',')
tickers = [ticker.strip().upper() for ticker in tickers]

# Date range selection
today = datetime.now()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(today - timedelta(days=365*5), today),
    min_value=datetime(2000, 1, 1),
    max_value=today
)

start_date, end_date = date_range

# Regime detection settings
st.sidebar.header("Regime Detection")
regime_method = st.sidebar.selectbox(
    "Detection Method",
    ["KMeans", "HMM"],
    index=0
)
regime_features = st.sidebar.multiselect(
    "Features for Regime Detection",
    ["returns", "volatility", "momentum", "trend"],
    default=["returns", "volatility", "momentum", "trend"]
)
lookback_window = st.sidebar.slider(
    "Lookback Window (days)",
    min_value=10,
    max_value=120,
    value=60,
    step=5
)

# Signal settings
st.sidebar.header("Signals")
momentum_periods = st.sidebar.multiselect(
    "Momentum Signal Periods",
    [5, 10, 20, 60, 120],
    default=[20, 60]
)
use_volatility_breakout = st.sidebar.checkbox("Use Volatility Breakout", value=True)
use_rsi_divergence = st.sidebar.checkbox("Use RSI Divergence", value=True)
use_ma_crossover = st.sidebar.checkbox("Use Moving Average Crossover", value=True)

# Portfolio settings
st.sidebar.header("Portfolio Construction")
rebalance_frequency = st.sidebar.slider(
    "Rebalance Frequency (days)",
    min_value=1,
    max_value=60,
    value=5,
    step=1
)
top_n_signals = st.sidebar.slider(
    "Top N Signals to Use",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)
transaction_cost = st.sidebar.slider(
    "Transaction Cost (%)",
    min_value=0.0,
    max_value=0.5,
    value=0.1,
    step=0.01
) / 100.0

# Run backtest button
run_backtest = st.sidebar.button("Run Backtest")

# Main app flow
if run_backtest:
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 1. Load data
    status_text.text("Loading market data...")
    data_loader = DataLoader()
    
    # Convert dates to string format for DataLoader
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Load data
    data_dict = data_loader.load_historical_data(
        tickers=tickers,
        start_date=start_date_str,
        end_date=end_date_str,
        interval='1d'
    )
    
    # Preprocess data
    processed_data = data_loader.preprocess_data(data_dict)
    
    # Create market index
    market_index = data_loader.create_market_index(processed_data)
    
    # Show first ticker data for regime detection
    main_ticker = tickers[0]
    main_data = processed_data[main_ticker]
    
    progress_bar.progress(20)
    
    # 2. Detect market regimes
    status_text.text("Detecting market regimes...")
    
    regime_detector = MarketRegimeDetector(
        method=regime_method.lower(),
        n_regimes=4,
        lookback_window=lookback_window,
        features=regime_features
    )
    
    # Fit regime detector and make predictions
    regime_detector.fit(main_data)
    data_with_regimes = regime_detector.predict_regimes(main_data)
    
    # Extract regime series
    regimes = data_with_regimes['regime']
    
    # Get regime statistics
    regime_stats = regime_detector.get_regime_statistics(data_with_regimes)
    
    progress_bar.progress(40)
    
    # 3. Create and backtest signals
    status_text.text("Generating trading signals...")
    
    # Create signals
    signals = []
    
    # Add momentum signals
    for period in momentum_periods:
        signals.append(MomentumSignal(lookback_period=period))
    
    # Add volatility breakout signal
    if use_volatility_breakout:
        signals.append(VolatilityBreakoutSignal(window=20, num_std=2.0))
        signals.append(VolatilityBreakoutSignal(window=10, num_std=1.5))
    
    # Add RSI divergence signal
    if use_rsi_divergence:
        signals.append(RSIDivergenceSignal(rsi_period=14, divergence_window=10))
    
    # Add MA crossover signals
    if use_ma_crossover:
        signals.append(MASignal(fast_period=50, slow_period=200))
        signals.append(MASignal(fast_period=20, slow_period=50))
    
    # Create backtester
    backtester = SignalBacktester(
        signals=signals,
        forward_returns_period=5,
        transaction_cost=transaction_cost
    )
    
    # Backtest signals by regime
    signal_performance_by_regime = backtester.backtest_signals_by_regime(data_with_regimes)
    
    # Generate signal features
    signal_features = backtester.generate_signal_features(main_data)
    
    progress_bar.progress(60)
    
    # 4. Build portfolio
    status_text.text("Constructing adaptive portfolio...")
    
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
        transaction_cost=transaction_cost
    )
    
    # Backtest portfolio
    portfolio_results = portfolio_constructor.backtest_portfolio(
        signal_returns=signal_returns,
        regimes=regimes.iloc[1:] if len(regimes) > 1 else None,
        rebalance_frequency=rebalance_frequency,
        top_n_signals=top_n_signals
    )
    
    progress_bar.progress(80)
    
    # 5. Calculate risk metrics and generate report
    status_text.text("Calculating performance metrics...")
    
    risk_manager = RiskManager()
    
    # Extract portfolio returns
    portfolio_returns = portfolio_results['portfolio_return']
    
    # Generate report
    performance_report = risk_manager.generate_report(
        portfolio_returns=portfolio_returns,
        benchmark_returns=main_data['Returns'].iloc[1:],
        regimes=regimes.iloc[1:] if len(regimes) > 1 else None
    )
    
    progress_bar.progress(100)
    status_text.text("Backtest complete!")
    
    # Display results
    st.header("Backtest Results")
    
    # 1. Performance Metrics
    st.subheader("Performance Metrics")
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = performance_report['metrics']
    
    col1.metric(
        "Total Return", 
        f"{metrics['total_return'] * 100:.2f}%"
    )
    
    col2.metric(
        "Annualized Return", 
        f"{metrics['annualized_return'] * 100:.2f}%"
    )
    
    col3.metric(
        "Sharpe Ratio", 
        f"{metrics['sharpe_ratio']:.2f}"
    )
    
    col4.metric(
        "Max Drawdown", 
        f"{metrics['max_drawdown'] * 100:.2f}%"
    )
    
    # 2. Market Regimes
    st.subheader("Market Regimes")
    
    # Plot regimes
    fig = regime_detector.visualize_regimes(data_with_regimes)
    st.pyplot(fig)
    
    # Display regime statistics
    st.dataframe(regime_stats)
    
    # 3. Signal Performance
    st.subheader("Signal Performance by Regime")
    
    # Create tabs for each regime
    regime_tabs = st.tabs([f"Regime {i}: {regime_detector.REGIME_NAMES.get(i, '')}" 
                         for i in range(len(signal_performance_by_regime))])
    
    for i, tab in enumerate(regime_tabs):
        with tab:
            if i in signal_performance_by_regime:
                tab.dataframe(signal_performance_by_regime[i])
            else:
                tab.write("No data for this regime")
    
    # 4. Portfolio Performance
    st.subheader("Portfolio Performance")
    
    # Plot portfolio value
    fig = portfolio_constructor.plot_backtest_results(portfolio_results)
    st.pyplot(fig)
    
    # 5. Drawdowns
    st.subheader("Drawdowns")
    
    # Plot drawdowns
    fig = risk_manager.plot_drawdowns(portfolio_returns)
    st.pyplot(fig)
    
    # 6. Stress Tests
    st.subheader("Stress Tests")
    
    # Display stress test results
    st.dataframe(performance_report['stress_tests'])
    
    # 7. Regime-Specific Performance
    if performance_report['regime_performance'] is not None:
        st.subheader("Performance by Market Regime")
        
        # Create DataFrame for regime performance
        regime_perf_data = []
        
        for regime, regime_metrics in performance_report['regime_performance'].items():
            row = {
                'Regime': f"{regime}: {regime_detector.REGIME_NAMES.get(regime, '')}",
                'Return (%)': regime_metrics['annualized_return'] * 100,
                'Volatility (%)': regime_metrics['volatility'] * 100,
                'Sharpe': regime_metrics['sharpe_ratio'],
                'Max Drawdown (%)': regime_metrics['max_drawdown'] * 100
            }
            regime_perf_data.append(row)
            
        regime_perf_df = pd.DataFrame(regime_perf_data)
        st.dataframe(regime_perf_df)

else:
    # Show instructions if backtest hasn't been run
    st.info("👈 Configure your backtest parameters in the sidebar and click 'Run Backtest' to start.")
    
    # Placeholder images
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Market Regimes")
        st.write("The system will detect different market regimes (Bull, Bear, Sideways, Volatile) using machine learning.")
        
    with col2:
        st.subheader("Adaptive Portfolio")
        st.write("Portfolio weights are dynamically adjusted based on which signals perform best in the current regime.")
        
    # Example workflow
    st.subheader("How It Works")
    st.markdown(
        """
        1. **Data Loading**: Historical market data is loaded for the selected tickers
        2. **Regime Detection**: Machine learning identifies distinct market states
        3. **Signal Generation**: Multiple alpha signals are calculated
        4. **Backtesting**: Signals are tested in each regime and ranked by performance
        5. **Portfolio Construction**: Capital is allocated to the best-performing signals
        6. **Performance Analysis**: Risk metrics and visualizations are generated
        """
    ) 