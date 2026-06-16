import pandas as pd
from scipy import stats


class NormalityResult:
    def __init__(self, statistic, p_value, is_normal, method):
        self.statistic = statistic
        self.p_value = p_value
        self.is_normal = is_normal
        self.method = method


def test_normality(series, alpha=0.05):
    data = pd.Series(series).dropna()

    if len(data) < 3:
        return NormalityResult(
            statistic=0.0,
            p_value=1.0,
            is_normal=True,
            method="insufficient_data",
        )

    if data.nunique() <= 1:
        return NormalityResult(
            statistic=0.0,
            p_value=1.0,
            is_normal=True,
            method="constant",
        )

    if len(data) <= 5000:
        statistic, p_value = stats.shapiro(data)
        method = "shapiro"
    else:
        statistic, p_value = stats.normaltest(data)
        method = "normaltest"

    return NormalityResult(
        statistic=float(statistic),
        p_value=float(p_value),
        is_normal=bool(p_value >= alpha),
        method=method,
    )
