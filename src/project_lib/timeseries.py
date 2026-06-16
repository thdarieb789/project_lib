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
        if hasattr(detector, "detector") and hasattr(detector, "series"):
            self.anomalies = detector.detector.detect(detector.series)
            return self.anomalies

        self.anomalies = detector.detect(self.data)
        return self.anomalies

    def find_anomalies_auto(self, alpha=0.05, verbose=True):
        from .analysis import select_detector_result

        selection = select_detector_result(self.data, alpha=alpha)
        self.anomalies = selection.detector.detect(selection.series)
        self.selection_result = selection
        self.selection_report = selection.report(anomalies=self.anomalies)

        if verbose:
            print(self.selection_report)

        return self.anomalies
