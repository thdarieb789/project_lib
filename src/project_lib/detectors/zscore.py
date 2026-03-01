# detectors/zscore.py
import pandas as pd
from .base import BaseDetector


class ZScoreDetector(BaseDetector):

    def __init__(self, threshold=3):
        self.threshold = threshold

    def detect(self, series):
        std = series.std()
        if std == 0:
            return pd.Series(False, index=series.index)

        z_scores = (series - series.mean()) / std
        return abs(z_scores) > self.threshold