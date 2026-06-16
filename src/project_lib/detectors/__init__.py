from .base import BaseDetector
from .chauvenet import ChauvenetCriteria
from .dixon import DixonQDetector
from .grubbs import GrubbsDetector
from .hampel import HampelDetector
from .iqr import IQRDetector
from .modified_zscore import ModifiedZScoreDetector
from .monte_carlo import MonteCarloDetector
from .zscore import ZScoreDetector

ChauvenetDetector = ChauvenetCriteria

__all__ = [
    "BaseDetector",
    "ChauvenetCriteria",
    "ChauvenetDetector",
    "DixonQDetector",
    "GrubbsDetector",
    "HampelDetector",
    "IQRDetector",
    "ModifiedZScoreDetector",
    "MonteCarloDetector",
    "ZScoreDetector",
]
