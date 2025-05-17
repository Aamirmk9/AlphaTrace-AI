# AlphaTrace AI

A Python-based real-time alpha generation and backtesting platform for quantitative finance that dynamically adjusts trading signals and portfolio weights based on market regimes (bull, bear, sideways, volatile).

## Features

- **Market Regime Detection**: Automatically identify distinct market states using machine learning
- **Multi-Signal Alpha Generation**: Engineer and test various trading signals
- **Regime-Specific Optimization**: Select the best-performing signals for each market regime
- **Adaptive Portfolio Construction**: Dynamically adjust portfolio weights based on the current market environment
- **Comprehensive Risk Analytics**: Calculate performance metrics and run stress tests
- **Interactive Interface**: Visualize results through a Streamlit web application

## Core Components

### 1. Market Regime Detection Module

Uses clustering algorithms (KMeans or Hidden Markov Models) to label market periods into distinct regimes:

- **Bull Market**: Strong uptrend, low volatility
- **Bear Market**: Downtrend, potentially high volatility
- **Sideways Market**: Range-bound, low momentum
- **Volatile Market**: High volatility, unclear direction

Features used for regime detection include returns, volatility, momentum indicators, and trend measures.

### 2. Signal Modeling Engine

Engineers multiple alpha signals and evaluates their performance in each market regime:

- **Momentum Signals**: Capture price trends over different time horizons
- **Volatility Breakout Signals**: Identify potential breakouts from price ranges
- **RSI Divergence Signals**: Detect divergences between price and momentum
- **Moving Average Crossover Signals**: Classic trend-following indicators

Signals are ranked by Sharpe ratio and other performance metrics within each regime.

### 3. Adaptive Portfolio Construction

Constructs portfolios that adapt to changing market conditions:

- **Regime-Based Signal Selection**: Allocates to top-performing signals in the current regime
- **Mean-Variance Optimization**: Finds optimal weights using classical portfolio theory
- **L2 Regularization**: Reduces portfolio turnover and transaction costs
- **Dynamic Rebalancing**: Adjusts allocations when regimes change

### 4. Risk & Backtest Framework

Provides comprehensive risk analytics and stress testing:

- **Performance Metrics**: Sharpe ratio, Sortino ratio, maximum drawdown, alpha/beta
- **Regime-Specific Analysis**: Compare performance across different market environments
- **Stress Testing**: Simulate portfolio behavior under extreme market conditions
- **Visualization**: Generate interactive charts for performance analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AlphaTrace.git
cd AlphaTrace

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Streamlit Web Application

Launch the interactive web interface:

```bash
streamlit run app.py
```

This will open a browser window where you can:
1. Configure data sources and parameters
2. Run backtests
3. Visualize regime detection and portfolio performance
4. Analyze risk metrics

### Programmatic Usage

You can also use AlphaTrace AI programmatically:

```python
# See example.py for a complete usage example
from data_utils import DataLoader
from regime_detection import MarketRegimeDetector
from signal_modeling import MomentumSignal, SignalBacktester
from portfolio import PortfolioConstructor
from risk_management import RiskManager

# Load and process data
data_loader = DataLoader()
data = data_loader.load_historical_data(["SPY"], "2018-01-01", "2023-01-01")
processed_data = data_loader.preprocess_data(data)

# Detect market regimes
regime_detector = MarketRegimeDetector()
regime_detector.fit(processed_data["SPY"])
data_with_regimes = regime_detector.predict_regimes(processed_data["SPY"])

# Generate and backtest signals
signals = [MomentumSignal(lookback_period=20)]
backtester = SignalBacktester(signals)
performance = backtester.backtest_all_signals(data_with_regimes)

# Construct portfolio
portfolio = PortfolioConstructor()
backtest = portfolio.backtest_portfolio(signal_returns)

# Analyze performance
risk_manager = RiskManager()
report = risk_manager.generate_report(backtest["portfolio_return"])
```

## Running Tests

To verify that all components are working correctly:

```bash
python test.py
```

## Project Structure

- **data_utils.py**: Data loading and preprocessing
- **regime_detection.py**: Market regime detection algorithms
- **signal_modeling.py**: Alpha signal generation and backtesting
- **portfolio.py**: Adaptive portfolio construction
- **risk_management.py**: Performance metrics and stress testing
- **app.py**: Streamlit web application
- **example.py**: Example script demonstrating usage
- **test.py**: Test script for validation

## Requirements

- Python 3.7+
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Plotly
- yfinance
- XGBoost
- CatBoost
- hmmlearn
- Streamlit
- SciPy

## Future Enhancements

- Integration with alternative data sources
- Reinforcement learning for weight optimization
- Real-time trading signals via API
- Factor analysis and risk decomposition
- Monte Carlo simulation for risk management

## License

MIT License 