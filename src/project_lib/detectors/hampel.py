import numpy as np
import pandas as pd

from .base import BaseDetector


class HampelDetector(BaseDetector):

    def __init__(self, window_size=3, n_sigmas=3):
        self.window_size = window_size
        self.n_sigmas = n_sigmas

    def detect(self, series):
        data = pd.Series(series)
        rolling_median = data.rolling(
            window=2 * self.window_size + 1,
            center=True,
            min_periods=1,
        ).median()

        deviation = (data - rolling_median).abs()
        mad = deviation.rolling(
            window=2 * self.window_size + 1,
            center=True,
            min_periods=1,
        ).median()

        threshold = self.n_sigmas * 1.4826 * mad
        return (deviation > threshold).fillna(False)
