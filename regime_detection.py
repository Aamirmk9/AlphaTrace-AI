import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from hmmlearn import hmm
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class MarketRegimeDetector:
    """
    Detects market regimes (bull, bear, sideways, volatile) using clustering algorithms.
    """
    
    REGIME_NAMES = {
        0: "Bull",       # Strong uptrend, low volatility
        1: "Bear",       # Downtrend, potentially high volatility
        2: "Sideways",   # Range-bound, low momentum
        3: "Volatile"    # High volatility, unclear direction
    }
    
    def __init__(self, method: str = "kmeans", n_regimes: int = 4, 
                 lookback_window: int = 20, features: Optional[List[str]] = None):
        """
        Initialize the regime detector.
        
        Args:
            method: Clustering method ('kmeans' or 'hmm')
            n_regimes: Number of market regimes to detect
            lookback_window: Window size for feature calculations
            features: List of features to use for clustering
        """
        self.method = method.lower()
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        
        # Default features if none provided
        self.features = features or ["returns", "volatility", "momentum", "trend"]
        
        # Initialize model
        self.model = None
        self.scaler = StandardScaler()
        
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features for regime detection.
        
        Args:
            df: DataFrame with financial time series data
            
        Returns:
            DataFrame with calculated features
        """
        feature_df = pd.DataFrame(index=df.index)
        
        # Return features
        if "returns" in self.features:
            feature_df["daily_returns"] = df["Returns"] if "Returns" in df.columns else df["Close"].pct_change()
            feature_df["rolling_returns"] = feature_df["daily_returns"].rolling(self.lookback_window).mean()
            
        # Volatility features
        if "volatility" in self.features:
            feature_df["volatility"] = feature_df.get("daily_returns", df["Close"].pct_change()).rolling(self.lookback_window).std()
            if "High" in df.columns and "Low" in df.columns:
                feature_df["range"] = (df["High"] - df["Low"]) / df["Close"].shift(1)
                feature_df["avg_range"] = feature_df["range"].rolling(self.lookback_window).mean()
            
        # Momentum features
        if "momentum" in self.features:
            # Calculate Rate of Change over different periods
            close_prices = df["Close"]
            feature_df["roc_5"] = (close_prices / close_prices.shift(5) - 1)
            feature_df["roc_10"] = (close_prices / close_prices.shift(10) - 1)
            feature_df["roc_20"] = (close_prices / close_prices.shift(20) - 1)
            
            # RSI-like momentum indicator
            returns = close_prices.pct_change()
            gains = returns.copy()
            losses = returns.copy()
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            
            avg_gain = gains.rolling(14).mean()
            avg_loss = losses.abs().rolling(14).mean()
            
            rs = avg_gain / avg_loss
            feature_df["rsi"] = 100 - (100 / (1 + rs))
            
        # Trend features
        if "trend" in self.features:
            # Simple moving averages
            feature_df["sma_20"] = df["Close"].rolling(20).mean()
            feature_df["sma_50"] = df["Close"].rolling(50).mean()
            
            # Moving average relationship
            if self.lookback_window >= 50:
                feature_df["ma_ratio"] = feature_df["sma_20"] / feature_df["sma_50"]
            
            # Linear regression slope (simplified approach)
            def rolling_slope(x):
                y = np.array(range(len(x)))
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                return np.sum((x - x_mean) * (y - y_mean)) / np.sum((y - y_mean) ** 2)
            
            feature_df["slope"] = df["Close"].rolling(self.lookback_window).apply(
                rolling_slope, raw=True)
            
        # Drop columns that are fully NaN
        feature_df = feature_df.dropna(axis=1, how='all')
            
        # Fill remaining NaNs
        feature_df = feature_df.fillna(method='bfill').fillna(0)
        
        return feature_df

    def _fit_kmeans(self, features: pd.DataFrame):
        """
        Fit KMeans clustering model.
        
        Args:
            features: Feature DataFrame
        """
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('kmeans', KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10))
        ])
        
        self.model.fit(features)
        
    def _fit_hmm(self, features: pd.DataFrame):
        """
        Fit Hidden Markov Model.
        
        Args:
            features: Feature DataFrame
        """
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Initialize HMM with Gaussian emissions
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
        
        self.model.fit(scaled_features)
        
    def fit(self, df: pd.DataFrame):
        """
        Fit the regime detection model.
        
        Args:
            df: DataFrame with financial time series data
        """
        # Calculate features
        features = self._calculate_features(df)
        
        # Train the appropriate model
        if self.method == "kmeans":
            self._fit_kmeans(features)
        elif self.method == "hmm":
            self._fit_hmm(features)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self

    def predict_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict market regimes for the given data.
        
        Args:
            df: DataFrame with financial time series data
            
        Returns:
            DataFrame with original data and regime labels
        """
        if self.model is None:
            raise ValueError("Model must be fit before predicting regimes")
            
        # Calculate features
        features = self._calculate_features(df)
        
        # Make predictions
        if self.method == "kmeans":
            regimes = self.model.predict(features)
        elif self.method == "hmm":
            scaled_features = self.scaler.transform(features)
            regimes = self.model.predict(scaled_features)
        
        # Create result DataFrame
        result = df.copy()
        result["regime"] = regimes
        result["regime_name"] = result["regime"].map(self.REGIME_NAMES)
        
        return result
    
    def visualize_regimes(self, df: pd.DataFrame, price_col: str = "Close", 
                          figsize: Tuple[int, int] = (12, 8)):
        """
        Visualize the detected market regimes.
        
        Args:
            df: DataFrame with regime labels
            price_col: Column name for price data
            figsize: Figure size
        """
        if "regime" not in df.columns:
            raise ValueError("DataFrame must contain regime labels. Run predict_regimes first.")
            
        plt.figure(figsize=figsize)
        
        # Plot price series
        ax1 = plt.subplot(211)
        for regime in range(self.n_regimes):
            regime_data = df[df["regime"] == regime]
            plt.plot(regime_data.index, regime_data[price_col], 'o-', markersize=2, 
                     label=f"Regime {regime}: {self.REGIME_NAMES.get(regime, '')}")
        
        plt.title("Market Regimes Detection")
        plt.ylabel("Price")
        plt.legend()
        
        # Plot regime changes over time
        ax2 = plt.subplot(212, sharex=ax1)
        plt.plot(df.index, df["regime"], 'k-')
        plt.ylabel("Regime")
        plt.xlabel("Date")
        plt.yticks(range(self.n_regimes), 
                   [f"{i}: {self.REGIME_NAMES.get(i, '')}" for i in range(self.n_regimes)])
        plt.grid(True)
        
        plt.tight_layout()
        
        return plt.gcf()
    
    def get_regime_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get statistics for each detected regime.
        
        Args:
            df: DataFrame with regime labels
            
        Returns:
            DataFrame with regime statistics
        """
        if "regime" not in df.columns:
            raise ValueError("DataFrame must contain regime labels. Run predict_regimes first.")
            
        stats = []
        
        for regime in range(self.n_regimes):
            regime_data = df[df["regime"] == regime]
            
            if len(regime_data) == 0:
                continue
                
            # Calculate statistics
            returns = regime_data["Returns"] if "Returns" in df.columns else regime_data["Close"].pct_change().dropna()
            
            stat = {
                "regime": regime,
                "regime_name": self.REGIME_NAMES.get(regime, f"Regime {regime}"),
                "count": len(regime_data),
                "avg_return": returns.mean(),
                "volatility": returns.std(),
                "sharpe": returns.mean() / returns.std() if returns.std() > 0 else 0,
                "max_return": returns.max(),
                "min_return": returns.min(),
                "skew": returns.skew(),
                "kurtosis": returns.kurtosis()
            }
            
            stats.append(stat)
            
        return pd.DataFrame(stats) 