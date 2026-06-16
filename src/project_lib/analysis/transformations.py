import numpy as np
import pandas as pd
from scipy import stats


class TransformationResult:
    def __init__(self, series, method, offset=0.0, lambda_=None):
        self.series = series
        self.method = method
        self.offset = offset
        self.lambda_ = lambda_


def _as_positive_series(series):
    data = pd.Series(series).astype(float)
    min_value = data.min()
    offset = 0.0

    if min_value <= 0:
        offset = abs(min_value) + 1.0

    return data + offset, offset


def log_transform(series):
    data, offset = _as_positive_series(series)
    return TransformationResult(
        series=np.log(data),
        method="log",
        offset=offset,
    )


def sqrt_transform(series):
    data, offset = _as_positive_series(series)
    return TransformationResult(
        series=np.sqrt(data),
        method="sqrt",
        offset=offset,
    )


def boxcox_transform(series):
    data, offset = _as_positive_series(series)

    if data.nunique() <= 1:
        return TransformationResult(
            series=pd.Series(0.0, index=data.index),
            method="boxcox",
            offset=offset,
            lambda_=None,
        )

    transformed, lambda_ = stats.boxcox(data)
    return TransformationResult(
        series=pd.Series(transformed, index=data.index),
        method="boxcox",
        offset=offset,
        lambda_=float(lambda_),
    )


def yeojohnson_transform(series):
    data = pd.Series(series).astype(float)

    if data.nunique() <= 1:
        return TransformationResult(
            series=pd.Series(0.0, index=data.index),
            method="yeo-johnson",
            lambda_=None,
        )

    transformed, lambda_ = stats.yeojohnson(data)
    return TransformationResult(
        series=pd.Series(transformed, index=data.index),
        method="yeo-johnson",
        lambda_=float(lambda_),
    )
