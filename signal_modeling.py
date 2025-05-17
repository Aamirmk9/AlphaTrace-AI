import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from typing import List, Dict, Tuple, Optional, Union
import warnings
from datetime import datetime, timedelta


class AlphaSignal:
    """
    Base class for alpha signals.
    """
    
    def __init__(self, name: str, parameters: Dict = None):
        """
        Initialize an alpha signal.
        
        Args:
            name: Signal name
            parameters: Dictionary of parameters for signal generation
        """
        self.name = name
        self.parameters = parameters or {}
        
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signal from data.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Series with signal values
        """
        raise NotImplementedError("Subclasses must implement generate()")
    
    def __str__(self) -> str:
        return f"{self.name}"


class MomentumSignal(AlphaSignal):
    """
    Price momentum signal.
    """
    
    def __init__(self, lookback_period: int = 20):
        """
        Initialize momentum signal.
        
        Args:
            lookback_period: Period for calculating momentum
        """
        super().__init__(
            name=f"Momentum_{lookback_period}d",
            parameters={"lookback_period": lookback_period}
        )
        
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate momentum signal (Rate of Change).
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Series with momentum values
        """
        if "Close" not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column")
            
        lookback = self.parameters["lookback_period"]
        signal = df["Close"].pct_change(lookback)
        
        return signal


class VolatilityBreakoutSignal(AlphaSignal):
    """
    Volatility breakout signal based on Bollinger Bands.
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        """
        Initialize volatility breakout signal.
        
        Args:
            window: Window size for calculating volatility
            num_std: Number of standard deviations for Bollinger Bands
        """
        super().__init__(
            name=f"BB_Width_{window}d_{num_std}std",
            parameters={"window": window, "num_std": num_std}
        )
        
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate Bollinger Band width signal.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Series with BB width values
        """
        if "Close" not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column")
            
        window = self.parameters["window"]
        num_std = self.parameters["num_std"]
        
        # Calculate Bollinger Bands
        rolling_mean = df["Close"].rolling(window=window).mean()
        rolling_std = df["Close"].rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        # BB width normalized by price
        bb_width = (upper_band - lower_band) / rolling_mean
        
        return bb_width


class RSIDivergenceSignal(AlphaSignal):
    """
    RSI divergence signal.
    """
    
    def __init__(self, rsi_period: int = 14, divergence_window: int = 10):
        """
        Initialize RSI divergence signal.
        
        Args:
            rsi_period: Period for calculating RSI
            divergence_window: Window for detecting divergence
        """
        super().__init__(
            name=f"RSI_Divergence_{rsi_period}d",
            parameters={"rsi_period": rsi_period, "divergence_window": divergence_window}
        )
        
    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """
        Calculate RSI.
        
        Args:
            prices: Series of prices
            
        Returns:
            Series with RSI values
        """
        delta = prices.diff()
        
        gain = delta.copy()
        loss = delta.copy()
        
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = -loss
        
        period = self.parameters["rsi_period"]
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate RSI divergence signal.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Series with divergence values
        """
        if "Close" not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column")
            
        # Calculate RSI
        rsi = self._calculate_rsi(df["Close"])
        
        # Calculate price and RSI slopes over divergence window
        window = self.parameters["divergence_window"]
        
        def rolling_slope(x):
            if len(x) < 2:
                return np.nan
            y = np.array(range(len(x)))
            return np.polyfit(y, x, 1)[0]
        
        price_slope = df["Close"].rolling(window).apply(rolling_slope, raw=True)
        rsi_slope = rsi.rolling(window).apply(rolling_slope, raw=True)
        
        # Divergence occurs when price and RSI slopes have opposite signs
        divergence = price_slope * rsi_slope
        
        # Normalize to -1 to 1 range
        max_abs = divergence.abs().max()
        if max_abs > 0:
            divergence = divergence / max_abs
        
        return divergence


class MASignal(AlphaSignal):
    """
    Moving Average crossover signal.
    """
    
    def __init__(self, fast_period: int = 50, slow_period: int = 200):
        """
        Initialize Moving Average crossover signal.
        
        Args:
            fast_period: Fast MA period
            slow_period: Slow MA period
        """
        super().__init__(
            name=f"MA_{fast_period}_{slow_period}",
            parameters={"fast_period": fast_period, "slow_period": slow_period}
        )
        
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate MA crossover signal.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Series with MA crossover values
        """
        if "Close" not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column")
            
        fast_period = self.parameters["fast_period"]
        slow_period = self.parameters["slow_period"]
        
        # Calculate moving averages
        fast_ma = df["Close"].rolling(window=fast_period).mean()
        slow_ma = df["Close"].rolling(window=slow_period).mean()
        
        # Calculate ratio of fast to slow MA
        signal = fast_ma / slow_ma - 1
        
        return signal


class SignalCombiner:
    """
    Combines multiple alpha signals to predict returns.
    """
    
    def __init__(self, model_type: str = "xgboost", parameters: Dict = None):
        """
        Initialize signal combiner.
        
        Args:
            model_type: ML model type ('xgboost' or 'catboost')
            parameters: Dictionary of model parameters
        """
        self.model_type = model_type.lower()
        self.parameters = parameters or {}
        self.model = None
        self.feature_importances = None
        
    def _create_model(self):
        """
        Create the ML model for signal combination.
        """
        if self.model_type == "xgboost":
            params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "reg:squarederror",
                "random_state": 42
            }
            params.update(self.parameters)
            
            self.model = xgb.XGBRegressor(**params)
            
        elif self.model_type == "catboost":
            params = {
                "iterations": 100,
                "max_depth": 6,
                "learning_rate": 0.05,
                "random_seed": 42,
                "verbose": False
            }
            params.update(self.parameters)
            
            self.model = CatBoostRegressor(**params)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
    def train(self, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5):
        """
        Train the model on signal features.
        
        Args:
            X: Feature DataFrame with signals
            y: Target Series (forward returns)
            cv_splits: Number of CV splits for time series validation
        """
        if self.model is None:
            self._create_model()
            
        # Handle missing values
        X = X.fillna(0)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        scores = []
        importances = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate score
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            scores.append({"mse": mse, "r2": r2})
            
            # Store feature importance
            if hasattr(self.model, "feature_importances_"):
                importances.append(pd.Series(
                    self.model.feature_importances_, index=X.columns))
            
        # Train final model on all data
        self.model.fit(X, y)
        
        # Store feature importances
        if hasattr(self.model, "feature_importances_"):
            self.feature_importances = pd.Series(
                self.model.feature_importances_, index=X.columns)
        
        # Average cross-validation scores
        avg_scores = {
            key: np.mean([s[key] for s in scores]) for key in scores[0]
        }
        
        return avg_scores
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict returns from signal features.
        
        Args:
            X: Feature DataFrame with signals
            
        Returns:
            Series with predicted returns
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
            
        # Handle missing values
        X = X.fillna(0)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return pd.Series(predictions, index=X.index)


class SignalBacktester:
    """
    Backtests signals and calculates performance metrics.
    """
    
    def __init__(self, 
                 signals: List[AlphaSignal] = None, 
                 forward_returns_period: int = 5,
                 transaction_cost: float = 0.001):
        """
        Initialize backtester.
        
        Args:
            signals: List of alpha signals to backtest
            forward_returns_period: Days for forward returns calculation
            transaction_cost: Transaction cost per trade
        """
        self.signals = signals or []
        self.forward_returns_period = forward_returns_period
        self.transaction_cost = transaction_cost
        self.signal_performances = {}
        
    def add_signal(self, signal: AlphaSignal):
        """
        Add a signal to the backtester.
        
        Args:
            signal: Alpha signal to add
        """
        self.signals.append(signal)
        
    def backtest_signal(self, signal: AlphaSignal, df: pd.DataFrame, 
                        regime_filter: Optional[Union[int, List[int]]] = None) -> Dict:
        """
        Backtest a single signal.
        
        Args:
            signal: Alpha signal to backtest
            df: DataFrame with price and optional regime data
            regime_filter: Regime(s) to filter by
            
        Returns:
            Dictionary with performance metrics
        """
        # Generate signal
        try:
            signal_values = signal.generate(df)
            
            # Filter by regime if specified
            if regime_filter is not None and "regime" in df.columns:
                if isinstance(regime_filter, int):
                    regime_filter = [regime_filter]
                    
                mask = df["regime"].isin(regime_filter)
                signal_values = signal_values[mask]
                df_regime = df[mask]
            else:
                df_regime = df
                
            # Calculate forward returns
            forward_returns = df_regime["Close"].pct_change(
                self.forward_returns_period).shift(-self.forward_returns_period)
                
            # Align signal and returns
            common_idx = signal_values.index.intersection(forward_returns.index)
            signal_values = signal_values.loc[common_idx]
            forward_returns = forward_returns.loc[common_idx]
            
            # Remove missing values
            mask = ~(signal_values.isna() | forward_returns.isna())
            signal_values = signal_values[mask]
            forward_returns = forward_returns[mask]
            
            if len(signal_values) == 0:
                return {
                    "signal": signal.name,
                    "count": 0,
                    "sharpe": np.nan,
                    "hit_rate": np.nan,
                    "avg_return": np.nan,
                    "annualized_return": np.nan,
                    "volatility": np.nan,
                    "max_drawdown": np.nan,
                    "correlation": np.nan
                }
                
            # Standardize signal
            signal_values = (signal_values - signal_values.mean()) / signal_values.std()
            
            # Calculate signal direction (sign)
            signal_signs = np.sign(signal_values)
            
            # Calculate returns based on signal direction
            strategy_returns = signal_signs * forward_returns - self.transaction_cost
            
            # Calculate performance metrics
            count = len(strategy_returns)
            avg_return = strategy_returns.mean()
            volatility = strategy_returns.std()
            
            # Annualize returns and volatility (assuming 252 trading days)
            ann_factor = 252 / self.forward_returns_period
            annualized_return = avg_return * ann_factor
            annualized_vol = volatility * np.sqrt(ann_factor)
            
            # Sharpe ratio
            sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
            
            # Hit rate
            hit_rate = (strategy_returns > 0).mean()
            
            # Correlation with forward returns
            correlation = signal_values.corr(forward_returns)
            
            # Maximum drawdown
            cum_returns = (1 + strategy_returns).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns / running_max) - 1
            max_drawdown = drawdown.min()
            
            return {
                "signal": signal.name,
                "count": count,
                "sharpe": sharpe,
                "hit_rate": hit_rate,
                "avg_return": avg_return,
                "annualized_return": annualized_return,
                "volatility": volatility,
                "max_drawdown": max_drawdown,
                "correlation": correlation
            }
            
        except Exception as e:
            warnings.warn(f"Error backtesting signal {signal.name}: {str(e)}")
            return {
                "signal": signal.name,
                "count": 0,
                "sharpe": np.nan,
                "hit_rate": np.nan,
                "avg_return": np.nan,
                "annualized_return": np.nan,
                "volatility": np.nan,
                "max_drawdown": np.nan,
                "correlation": np.nan
            }
    
    def backtest_all_signals(self, df: pd.DataFrame, 
                             regime_filter: Optional[Union[int, List[int]]] = None) -> pd.DataFrame:
        """
        Backtest all signals.
        
        Args:
            df: DataFrame with price and optional regime data
            regime_filter: Regime(s) to filter by
            
        Returns:
            DataFrame with performance metrics for all signals
        """
        results = []
        
        for signal in self.signals:
            result = self.backtest_signal(signal, df, regime_filter)
            results.append(result)
            
        return pd.DataFrame(results).sort_values("sharpe", ascending=False)
    
    def backtest_signals_by_regime(self, df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        Backtest signals for each regime.
        
        Args:
            df: DataFrame with price and regime data
            
        Returns:
            Dictionary mapping regime IDs to performance DataFrames
        """
        if "regime" not in df.columns:
            raise ValueError("DataFrame must contain 'regime' column")
            
        regimes = df["regime"].unique()
        results = {}
        
        for regime in regimes:
            regime_results = self.backtest_all_signals(df, regime_filter=regime)
            results[regime] = regime_results
            
        return results
    
    def generate_signal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features from all signals.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with signal features
        """
        features = pd.DataFrame(index=df.index)
        
        for signal in self.signals:
            try:
                signal_values = signal.generate(df)
                features[signal.name] = signal_values
            except Exception as e:
                warnings.warn(f"Error generating signal {signal.name}: {str(e)}")
                
        return features 