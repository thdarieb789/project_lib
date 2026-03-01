import numpy as np
import pandas as pd
from scipy import stats
from .base import BaseDetector


class GrubbsDetector(BaseDetector):

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def detect(self, series):
        n = len(series)
        if n < 3:
            return pd.Series(False, index=series.index)

        mean = series.mean()
        std = series.std()

        if std == 0:
            return pd.Series(False, index=series.index)

        abs_diff = abs(series - mean)
        max_deviation = abs_diff.max()
        max_index = abs_diff.idxmax()

        G = max_deviation / std

        t = stats.t.ppf(1 - self.alpha / (2 * n), n - 2)
        threshold = ((n - 1) / np.sqrt(n)) * np.sqrt(t**2 / (n - 2 + t**2))

        anomalies = pd.Series(False, index=series.index)
        if G > threshold:
            anomalies.loc[max_index] = True

        return anomalies