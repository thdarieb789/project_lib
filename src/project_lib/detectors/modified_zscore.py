import numpy as np
import pandas as pd

from .base import BaseDetector


class ModifiedZScoreDetector(BaseDetector):

    def __init__(self, threshold=3.5):
        self.threshold = threshold

    def detect(self, series):
        data = pd.Series(series)
        median = data.median()
        mad = np.median(np.abs(data - median))

        if mad == 0:
            return pd.Series(False, index=data.index)

        modified_z_scores = 0.6745 * (data - median) / mad
        return (modified_z_scores.abs() > self.threshold).fillna(False)
