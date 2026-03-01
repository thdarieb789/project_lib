import pandas as pd
import numpy as np


class TimeSeries:
    def __init__(self, data, time_index=None):
        self.data = self._convert_to_series(data, time_index)
        self.anomalies = pd.Series(False, index=self.data.index)
        self._validate()

    def _convert_to_series(self, data, time_index):
        if isinstance(data, pd.Series):
            return data

        elif isinstance(data, (list, np.ndarray)):
            if time_index is None:
                return pd.Series(data)
            return pd.Series(data, index=time_index)

        raise TypeError("Неподдерживаемый тип данных")

    def _validate(self):
        if self.data.isnull().any():
            raise ValueError("Временной ряд содержит значения NaN")

    def find_anomalies(self, detector):
        self.anomalies = detector.detect(self.data)
        return self.anomalies
