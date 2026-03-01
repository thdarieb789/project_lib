from .timeseries import TimeSeries
from .detectors import (
    BaseDetector,
    ChauvenetCriteria,
    ZScoreDetector,
    IQRDetector,
    DixonQDetector,
    GrubbsDetector,
)

__all__ = [
    "TimeSeries",
    "BaseDetector",
    "ChauvenetDetector",
    "ZScoreDetector",
    "IQRDetector",
    "GrubbsDetector",
    "DixonQDetector",
]

__version__ = "0.1.4"