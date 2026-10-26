from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd


class DistributionBinner:
    """Encapsulates the logic for binning a statistical distribution into discrete levels.

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
        spread_method: Literal[
            "std", "mad", "saturating", "asymmetric_saturating"
        ] = "std",
        max_z: float = 4.0,
        n_bins: int = 30,
        saturation_quantile: float = 0.98,
    ):
        """Initialize the DistributionBinner.

        Args:
            data: A pandas Series (index=keys, values=data) or DataFrame.
            value_column: If data is a DataFrame, the column name to use.
            center_method: Method to calculate the center ('mean', 'median', 'zero').
            spread_method: Method to calculate the spread:
                'std'  - one standard deviation per z-unit;
                'mad'  - one robust standard deviation per z-unit;
                'saturating' - stretch the palette over the data's own range, so
                    the `saturation_quantile` of |value - center| lands on the last
                    bin. Use this when the palette must *show* the spread that
                    exists rather than measure it in standard deviations: a
                    distribution whose values all sit within a fraction of one
                    standard deviation is otherwise painted entirely in the
                    neutral centre colour, which reads as "no signal anywhere".
                'asymmetric_saturating' - as above, but measure the lower and
                    upper halves independently. This is appropriate for skewed,
                    bounded magnitudes: an unattainable range below the data must
                    not consume the low-colour half of a sequential palette.
            max_z: The z-score value that corresponds to the maximum bin.
            n_bins: Number of bins for each direction (positive/negative).
            saturation_quantile: Quantile of |value - center| mapped to the last
                bin under the 'saturating' method. Values past it clip, so a
                single extreme outlier cannot flatten every other colour.
        """
        self.max_z = max_z
        self.n_bins = n_bins
        self.saturation_quantile = saturation_quantile

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
        self.spread_method = spread_method
        if spread_method == "asymmetric_saturating":
            self.lower_spread, self.upper_spread = self._calculate_asymmetric_spreads()
            # Preserve the legacy scalar audit value. It remains the furthest
            # saturation distance, while binning uses the side-specific values.
            self.spread = max(self.lower_spread, self.upper_spread)
        else:
            self.spread = self._calculate_spread(spread_method)
            self.lower_spread = self.spread
            self.upper_spread = self.spread

        # Pre-calculate bins
        self.bins = self._calculate_bins()

    @property
    def color_limit(self) -> float:
        """Furthest distance from the centre at which the palette saturates."""
        return max(self.lower_color_limit, self.upper_color_limit)

    @property
    def lower_color_limit(self) -> float:
        """Distance below the centre at which the low palette end saturates."""
        return float(self.max_z * self.lower_spread)

    @property
    def upper_color_limit(self) -> float:
        """Distance above the centre at which the high palette end saturates."""
        return float(self.max_z * self.upper_spread)

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
        if method == "saturating":
            deviations = (self.series - self.center).abs().dropna()
            if deviations.empty:
                return 0.0
            limit = float(np.nanquantile(deviations, self.saturation_quantile))
            if limit <= 0:
                # The chosen quantile is degenerate (a heavily tied distribution);
                # fall back to the widest deviation so colours still separate.
                limit = float(deviations.max())
            return limit / self.max_z if limit > 0 else 0.0
        if method == "mad":
            # Median Absolute Deviation
            # We scale by 1.4826 to make it consistent with sigma for normal distributions
            median = self.series.median()
            mad = (self.series - median).abs().median()
            return mad * 1.4826
        # std
        return self.series.std()

    def _calculate_asymmetric_spreads(self) -> tuple[float, float]:
        """Return independent robust spreads below and above ``center``."""
        values = self.series.dropna()
        lower_deviations = (self.center - values[values < self.center]).dropna()
        upper_deviations = (values[values > self.center] - self.center).dropna()

        def spread_for(deviations: pd.Series) -> float:
            if deviations.empty:
                return 0.0
            limit = float(np.nanquantile(deviations, self.saturation_quantile))
            if limit <= 0:
                limit = float(deviations.max())
            return limit / self.max_z if limit > 0 else 0.0

        return spread_for(lower_deviations), spread_for(upper_deviations)

    def _calculate_bins(self) -> pd.Series:
        """Calculate bin indices for all elements."""
        if self.spread_method == "asymmetric_saturating":
            # A bounded magnitude can have a short lower tail and a long upper
            # tail. Scaling both with the longer one turns its real low values
            # yellow before the palette ever reaches white.
            z_scores = pd.Series(np.nan, index=self.series.index, dtype=float)
            finite = self.series.notna()
            lower = finite & (self.series < self.center)
            upper = finite & (self.series > self.center)
            equal = finite & ~(lower | upper)
            lower_spread = self.lower_spread if self.lower_spread != 0 else 1.0
            upper_spread = self.upper_spread if self.upper_spread != 0 else 1.0
            z_scores.loc[lower] = (
                self.series.loc[lower] - self.center
            ) / lower_spread
            z_scores.loc[upper] = (
                self.series.loc[upper] - self.center
            ) / upper_spread
            z_scores.loc[equal] = 0.0
        else:
            # Avoid division by zero
            spread = self.spread if self.spread != 0 else 1.0
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
        """Returns the bin index for a given key.

        Args:
            key: The index key from the original data.

        Returns:
            int: The bin index. Returns 0 if key is not found.
        """
        try:
            return self.bins.loc[key]
        except KeyError:
            return 0

    def bin_for_value(self, value: float) -> int:
        """The bin a value falls in, as a registered element's does (``_calculate_bins``).

        For a value outside the distribution, such as a transporter family's Delta
        score on its subsystems' scale; a non-finite value is neutral, bin 0.
        """
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0
        if not np.isfinite(value):
            return 0
        if self.spread_method == "asymmetric_saturating":
            if value < self.center:
                z_score = (value - self.center) / (self.lower_spread if self.lower_spread != 0 else 1.0)
            elif value > self.center:
                z_score = (value - self.center) / (self.upper_spread if self.upper_spread != 0 else 1.0)
            else:
                z_score = 0.0
        else:
            z_score = (value - self.center) / (self.spread if self.spread != 0 else 1.0)
        if not np.isfinite(z_score):
            return 0
        step = self.max_z / self.n_bins
        return int(np.clip(np.round(z_score / step), -self.n_bins, self.n_bins))

    def get_all_bins(self) -> pd.Series:
        """Returns the Series of all bin indices."""
        return self.bins

    def get_histogram_data(self) -> list[dict[str, Any]]:
        """Returns histogram data including counts per bin and bin boundaries.

        Returns:
            List of dictionaries, each containing:
            - bin: The integer bin index
            - count: Number of items in this bin
            - label: A string description of the bin range (Z-score based)
            - value_center: The center value of the bin in original units
            - value_start: The start value of the bin in original units
            - value_end: The end value of the bin in original units
        """
        if self.spread_method == "asymmetric_saturating":
            return self._get_equal_width_histogram_data()

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
            val_start = self._value_for_z(z_start)
            val_end = self._value_for_z(z_end)
            val_center = self._value_for_z(z_center)

            histogram_bins.append({
                "bin": i,
                "count": count,
                "label": label,
                "value_center": val_center,
                "value_start": val_start,
                "value_end": val_end
            })

        return histogram_bins

    def _get_equal_width_histogram_data(self) -> list[dict[str, Any]]:
        """Build count bars with equal widths across an asymmetric color scale.

        Color bins intentionally have different numeric widths on either side of
        the median. Drawing those bins directly with a count axis would make bar
        area depend on width, so the hover chart uses equal numeric intervals and
        resolves each interval's color separately.
        """
        bin_count = 2 * self.n_bins + 1
        lower = self.center - self.lower_color_limit
        upper = self.center + self.upper_color_limit
        if lower == upper:
            lower -= 0.5
            upper += 0.5

        edges = np.linspace(lower, upper, bin_count + 1)
        finite = self.series.dropna().to_numpy(dtype=float)
        clipped = np.clip(finite, lower, upper)
        counts, _ = np.histogram(clipped, bins=edges)

        histogram_bins = []
        for position, count in enumerate(counts):
            value_start = float(edges[position])
            value_end = float(edges[position + 1])
            value_center = (value_start + value_end) / 2
            histogram_bins.append(
                {
                    "bin": position - self.n_bins,
                    "color_bin": self._bin_for_value(value_center),
                    "count": int(count),
                    "label": f"{value_start:.3g} to {value_end:.3g}",
                    "value_center": value_center,
                    "value_start": value_start,
                    "value_end": value_end,
                }
            )
        return histogram_bins

    def bin_for_value(self, value: float) -> int:
        """The palette bin an arbitrary value on this scale falls in."""
        return self._bin_for_value(value)

    def _bin_for_value(self, value: float) -> int:
        """Return the palette bin for an arbitrary value on this scale."""
        if not np.isfinite(value):
            return 0
        if value < self.center:
            spread = self.lower_spread
        elif value > self.center:
            spread = self.upper_spread
        else:
            return 0
        if spread == 0:
            return 0
        step = self.max_z / self.n_bins
        z_score = (value - self.center) / spread
        return int(np.clip(np.rint(z_score / step), -self.n_bins, self.n_bins))

    def _value_for_z(self, z_score: float) -> float:
        """Convert one standardized coordinate using the matching side's scale."""
        spread = self.lower_spread if z_score < 0 else self.upper_spread
        return z_score * spread + self.center

    def get_value(self, key: Any) -> Optional[float]:
        """Returns the original value for a given key."""
        if key in self.series.index:
            return self.series[key]
        return None
