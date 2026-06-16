import numpy as np
import pandas as pd

from .base import BaseDetector


class MonteCarloDetector(BaseDetector):
    """Monte Carlo outlier detector based on an extremeness statistic.

    H0: the sample comes from F without outliers.
    H1: the sample contains at least one outlier.

    The default statistic is T = max(abs(z_i)), where z_i is computed with the
    sample mean and sample standard deviation.
    """

    def __init__(
        self,
        simulations=10000,
        alpha=0.05,
        random_state=None,
        statistic="max_abs_zscore",
    ):
        self.simulations = simulations
        self.alpha = alpha
        self.random_state = random_state
        self.statistic = statistic
        self.last_critical_value = None
        self.last_observed_statistic = None
        self.last_p_values = None
        self.last_simulated_statistics = None

    def detect(self, series):
        data = self._validate_series(series)

        if len(data) < 3 or data.std(ddof=1) == 0:
            return pd.Series(False, index=data.index)

        simulated_statistics = self.simulate_null_distribution(data)
        critical_value = self.critical_value(simulated_statistics)
        statistic_values = self.statistic_values(data)
        observed_statistic = float(statistic_values.max())
        p_values = self.p_values(statistic_values, simulated_statistics)

        self.last_critical_value = critical_value
        self.last_observed_statistic = observed_statistic
        self.last_p_values = p_values
        self.last_simulated_statistics = simulated_statistics

        return pd.Series(
            (statistic_values > critical_value) & (p_values < self.alpha),
            index=data.index,
        )

    def statistic_values(self, series):
        data = self._validate_series(series)
        mean = data.mean()

        if self.statistic == "max_abs_deviation":
            return (data - mean).abs()

        if self.statistic == "max_abs_zscore":
            std = data.std(ddof=1)
            if std == 0:
                return pd.Series(0.0, index=data.index)
            return ((data - mean) / std).abs()

        raise ValueError("Неизвестная статистика экстремальности")

    def observed_statistic(self, series):
        return float(self.statistic_values(series).max())

    def simulate_null_distribution(self, series):
        data = self._validate_series(series)
        n = len(data)
        std = data.std(ddof=1)

        if n < 3 or std == 0:
            return np.zeros(self.simulations)

        rng = np.random.default_rng(self.random_state)
        samples = rng.normal(
            loc=data.mean(),
            scale=std,
            size=(self.simulations, n),
        )

        sample_means = samples.mean(axis=1, keepdims=True)

        if self.statistic == "max_abs_deviation":
            return np.max(np.abs(samples - sample_means), axis=1)

        if self.statistic == "max_abs_zscore":
            sample_stds = samples.std(axis=1, ddof=1, keepdims=True)
            z_scores = (samples - sample_means) / sample_stds
            return np.max(np.abs(z_scores), axis=1)

        raise ValueError("Неизвестная статистика экстремальности")

    def critical_value(self, simulated_statistics):
        return float(np.quantile(simulated_statistics, 1 - self.alpha))

    def p_values(self, statistic_values, simulated_statistics):
        values = pd.Series(statistic_values)
        p_values = [
            np.mean(simulated_statistics >= value)
            for value in values
        ]
        return pd.Series(p_values, index=values.index)

    def _validate_series(self, series):
        data = pd.Series(series)

        if data.isnull().any():
            raise ValueError("Временной ряд содержит значения NaN")

        if not pd.api.types.is_numeric_dtype(data):
            raise TypeError("Временной ряд должен содержать числовые значения")

        return data.astype(float)
