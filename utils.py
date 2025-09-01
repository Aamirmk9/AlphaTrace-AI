import numpy as np
import pandas as pd


def build_signal_returns(signal_features: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
    """
    Construct per-signal daily returns from raw signal features and price returns.
    Signals are standardized safely and aligned to the returns index.
    """
    output_frame = pd.DataFrame(index=signal_features.index[1:])
    for column_name in signal_features.columns:
        signal_series = signal_features[column_name].dropna()
        if signal_series.empty:
            continue
        standard_deviation = signal_series.std()
        if standard_deviation and standard_deviation > 0 and not np.isnan(standard_deviation):
            signal_series = (signal_series - signal_series.mean()) / standard_deviation
        else:
            # Skip degenerate signals
            continue
        aligned_direction = np.sign(signal_series.shift(1)).reindex(output_frame.index)
        output_frame[column_name] = aligned_direction * returns.reindex(output_frame.index)
    return output_frame.dropna(how="all", axis=1)


