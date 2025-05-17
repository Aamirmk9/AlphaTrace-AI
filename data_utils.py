import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, timedelta


class DataLoader:
    """
    Data loading and preprocessing utility for financial time series data.
    """
    
    def __init__(self, api_key: Optional[str] = None, data_source: str = "yfinance"):
        """
        Initialize the data loader.
        
        Args:
            api_key: API key for data providers like Polygon.io (not needed for yfinance)
            data_source: Source of data ("yfinance" or "polygon")
        """
        self.api_key = api_key
        self.data_source = data_source.lower()
        
    def load_historical_data(
        self, 
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "1d",
        include_fundamentals: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Load historical OHLCV data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (defaults to today)
            interval: Data frequency ('1d', '1wk', '1mo')
            include_fundamentals: Whether to include fundamental data
            
        Returns:
            Dictionary mapping tickers to their respective dataframes
        """
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        data_dict = {}
        
        if self.data_source == "yfinance":
            for ticker in tickers:
                try:
                    ticker_obj = yf.Ticker(ticker)
                    df = ticker_obj.history(start=start_date, end=end_date, interval=interval)
                    
                    # Standardize column names
                    df.columns = [col if col != "Close" else "Adj Close" if "Adj Close" in df.columns else col 
                                 for col in df.columns]
                    
                    # Ensure standard columns exist
                    required_cols = ["Open", "High", "Low", "Close", "Volume"]
                    for col in required_cols:
                        if col not in df.columns:
                            df[col] = np.nan
                    
                    # Add fundamentals if requested
                    if include_fundamentals:
                        # This is simplified - in a real implementation, you'd add actual fundamental data
                        quarterly_financials = ticker_obj.quarterly_financials
                        if not quarterly_financials.empty:
                            # Just as an example - in reality, you'd need to align dates properly
                            df['PE_Ratio'] = np.nan  
                            
                    data_dict[ticker] = df
                    
                except Exception as e:
                    print(f"Error loading data for {ticker}: {e}")
                    
        elif self.data_source == "polygon":
            # Placeholder for Polygon.io implementation
            # Would require the polygon-api-client library and proper API calls
            pass
            
        return data_dict
    
    def preprocess_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Preprocess data for all tickers.
        
        Args:
            data: Dictionary of dataframes
            
        Returns:
            Dictionary of preprocessed dataframes
        """
        processed_data = {}
        
        for ticker, df in data.items():
            # Handle missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Calculate basic features
            if 'Close' in df.columns and 'Open' in df.columns:
                df['Returns'] = df['Close'].pct_change()
                df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
                df['Range'] = (df['High'] - df['Low']) / df['Open']
                
                # Volatility (20-day rolling standard deviation of returns)
                df['Volatility_20d'] = df['Returns'].rolling(window=20).std()
                
            processed_data[ticker] = df.dropna()
            
        return processed_data
    
    def create_market_index(self, data: Dict[str, pd.DataFrame], weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Create a market index from multiple tickers.
        
        Args:
            data: Dictionary of ticker dataframes
            weights: Dictionary mapping tickers to their weights (equal by default)
            
        Returns:
            Dataframe containing the market index
        """
        if not data:
            raise ValueError("No data provided to create market index")
            
        tickers = list(data.keys())
        
        # Default to equal weighting
        if weights is None:
            weights = {ticker: 1.0 / len(tickers) for ticker in tickers}
            
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Align all dataframes to common dates
        aligned_returns = pd.DataFrame()
        
        for ticker in tickers:
            if ticker not in weights:
                continue
                
            if 'Returns' not in data[ticker].columns:
                data[ticker]['Returns'] = data[ticker]['Close'].pct_change()
                
            ticker_returns = data[ticker]['Returns']
            aligned_returns[ticker] = ticker_returns
            
        # Handle missing data
        aligned_returns = aligned_returns.fillna(0)
        
        # Create weighted index
        weighted_returns = pd.Series(0.0, index=aligned_returns.index)
        for ticker in tickers:
            if ticker in weights and ticker in aligned_returns.columns:
                weighted_returns += aligned_returns[ticker] * weights[ticker]
                
        # Calculate index level from returns
        index_level = (1 + weighted_returns).cumprod() * 100
        
        market_index = pd.DataFrame({
            'Index_Level': index_level,
            'Returns': weighted_returns
        })
        
        return market_index 