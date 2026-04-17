import numpy as np
import pandas as pd
from typing import Union, Optional, Literal, Any

class DistributionBinner:
    """
    Encapsulates the logic for binning a statistical distribution into discrete levels.
    
    This class takes a distribution of values (e.g., from a pandas Series or DataFrame),
    calculates statistics (center, spread), and maps each value to a bin index based on
    its Z-score (or robust Z-score).
    
    Bin Indices:
        0: Neutral / Center (or exact 0 if no threshold)
        +1 to +N: Positive bins of increasing intensity
        -1 to -N: Negative bins of increasing intensity
    """

    def __init__(
        self,
        data: Union[pd.Series, pd.DataFrame],
        value_column: Optional[str] = None,
        center_method: Literal["mean", "median", "zero"] = "mean",
        spread_method: Literal["std", "mad"] = "std",
        max_z: float = 4.0,
        n_bins: int = 30
    ):
        """
        Initialize the DistributionBinner.

        Args:
            data: A pandas Series (index=keys, values=data) or DataFrame.
            value_column: If data is a DataFrame, the column name to use.
            center_method: Method to calculate the center ('mean', 'median', 'zero').
            spread_method: Method to calculate the spread ('std', 'mad').
            max_z: The z-score value that corresponds to the maximum bin.
            n_bins: Number of bins for each direction (positive/negative).
        """
        self.max_z = max_z
        self.n_bins = n_bins
        
        # Prepare data
        if isinstance(data, pd.DataFrame):
            if value_column is None:
                raise ValueError("value_column must be provided if data is a DataFrame")
            self.series = data[value_column]
        else:
            self.series = data

        # Ensure series is numeric
        self.series = pd.to_numeric(self.series, errors='coerce')

        # Calculate stats
        self.center = self._calculate_center(center_method)
        self.spread = self._calculate_spread(spread_method)

        # Pre-calculate bins
        self.bins = self._calculate_bins()

    def _calculate_center(self, method: str) -> float:
        """Calculate the center of the distribution."""
        if method == "zero":
            return 0.0
        elif method == "median":
            return self.series.median()
        else: # mean
            return self.series.mean()

    def _calculate_spread(self, method: str) -> float:
        """Calculate the spread (denominator for Z-score) of the distribution."""
        if method == "mad":
            # Median Absolute Deviation
            # We scale by 1.4826 to make it consistent with sigma for normal distributions
            median = self.series.median()
            mad = (self.series - median).abs().median()
            return mad * 1.4826
        else: # std
            return self.series.std()

    def _calculate_bins(self) -> pd.Series:
        """Calculate bin indices for all elements."""
        # Avoid division by zero
        spread = self.spread if self.spread != 0 else 1.0
        
        # Calculate Z-scores
        z_scores = (self.series - self.center) / spread
        
        # Calculate bin step size
        step = self.max_z / self.n_bins
        
        # Initialize result series with 0 (Neutral)
        # This ensures NaNs are treated as neutral without being involved in calculation
        final_bins = pd.Series(0, index=self.series.index, dtype=int)
        
        # Process only valid (non-NaN) Z-scores
        valid_mask = z_scores.notna()
        
        if valid_mask.any():
            valid_z = z_scores[valid_mask]
            
            # Calculate bins using rounding (nearest neighbor)
            # This centers bin 0 at 0 with width 'step'
            # e.g. if step=1:
            # -0.5 <= z < 0.5 -> bin 0
            # 0.5 <= z < 1.5 -> bin 1
            bins = (valid_z / step).round().astype(int)
            
            # Clip to max bins
            bins = bins.clip(lower=-self.n_bins, upper=self.n_bins)
            
            # Assign calculated bins to valid positions
            final_bins[valid_mask] = bins
            
        return final_bins

    def get_bin(self, key: Any) -> int:
        """
        Returns the bin index for a given key.
        
        Args:
            key: The index key from the original data.
            
        Returns:
            int: The bin index. Returns 0 if key is not found.
        """
        try:
            return self.bins.loc[key]
        except KeyError:
            return 0

    def get_all_bins(self) -> pd.Series:
        """Returns the Series of all bin indices."""
        return self.bins

    def get_histogram_data(self) -> list[dict[str, Any]]:
        """
        Returns histogram data including counts per bin and bin boundaries.
        
        Returns:
            List of dictionaries, each containing:
            - bin: The integer bin index
            - count: Number of items in this bin
            - label: A string description of the bin range (Z-score based)
            - value_center: The center value of the bin in original units
            - value_start: The start value of the bin in original units
            - value_end: The end value of the bin in original units
        """
        # Count occurrences of each bin
        counts = self.bins.value_counts().to_dict()
        
        # Calculate step size for Z-scores
        step = self.max_z / self.n_bins
        
        histogram_bins = []
        
        # Iterate from -n_bins to +n_bins
        for i in range(-self.n_bins, self.n_bins + 1):
            count = counts.get(i, 0)
            
            # Calculate Z-score boundaries for this bin
            # Bin i covers [ (i-0.5)*step, (i+0.5)*step ]
            z_center = i * step
            z_start = (i - 0.5) * step
            z_end = (i + 0.5) * step
            
            # Calculate range labels
            if i == 0:
                label = "Neutral"
            elif i == self.n_bins:
                label = f"Z > {z_start:.1f}"
            elif i == -self.n_bins:
                label = f"Z < {z_end:.1f}"
            else:
                label = f"{z_start:.1f} < Z <= {z_end:.1f}"

            # Calculate original values
            val_start = z_start * self.spread + self.center
            val_end = z_end * self.spread + self.center
            val_center = z_center * self.spread + self.center

            histogram_bins.append({
                "bin": i,
                "count": count,
                "label": label,
                "value_center": val_center,
                "value_start": val_start,
                "value_end": val_end
            })
            
        return histogram_bins

    def get_value(self, key: Any) -> Optional[float]:
        """Returns the original value for a given key."""
        if key in self.series.index:
            return self.series[key]
        return None
