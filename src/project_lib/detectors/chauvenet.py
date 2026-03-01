import numpy as np
import pandas as pd
from scipy.stats import norm
from .base import BaseDetector


class ChauvenetCriteria(BaseDetector):

    def detect(self, series: pd.Series) -> pd.Series:
        n = len(series)
        if n < 2:
            return pd.Series(False, index=series.index)

        mean = series.mean()
        std = series.std()

        if std == 0:
            return pd.Series(False, index=series.index)

        k = norm.ppf(1 - 1 / (2 * n))
        anomalies = np.abs(series - mean) > k * std

        return pd.Series(anomalies, index=series.index).fillna(False)