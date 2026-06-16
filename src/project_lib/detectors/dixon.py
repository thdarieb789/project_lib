import pandas as pd
from .base import BaseDetector


class DixonQDetector(BaseDetector):

    def __init__(self, q_threshold=0.5):
        self.q_threshold = q_threshold

    def detect(self, series):
        if len(series) < 3:
            return pd.Series(False, index=series.index)

        sorted_series = series.sort_values()

        low_gap = sorted_series.iloc[1] - sorted_series.iloc[0]
        high_gap = sorted_series.iloc[-1] - sorted_series.iloc[-2]
        range_ = sorted_series.iloc[-1] - sorted_series.iloc[0]

        low_q = low_gap / range_ if range_ != 0 else 0
        high_q = high_gap / range_ if range_ != 0 else 0

        anomalies = pd.Series(False, index=series.index)

        if low_q > self.q_threshold:
            anomalies.loc[sorted_series.index[0]] = True

        if high_q > self.q_threshold:
            anomalies.loc[sorted_series.index[-1]] = True

        return anomalies
