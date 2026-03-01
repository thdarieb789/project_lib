import pandas as pd
from .base import BaseDetector


class IQRDetector(BaseDetector):

    def __init__(self, multiplier=1.5):
        self.multiplier = multiplier

    def detect(self, series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - self.multiplier * iqr
        upper = q3 + self.multiplier * iqr

        return (series < lower) | (series > upper)