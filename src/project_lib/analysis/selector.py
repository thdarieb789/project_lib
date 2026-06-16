import numpy as np
import pandas as pd
from scipy import stats
from project_lib.analysis.visualization import (plot_anomalies)

from project_lib.detectors import (
    ChauvenetDetector,
    DixonQDetector,
    GrubbsDetector,
    HampelDetector,
    IQRDetector,
    ModifiedZScoreDetector,
    MonteCarloDetector,
    ZScoreDetector,
)

from project_lib.analysis.normality import test_normality
from .transformations import (
    boxcox_transform,
    log_transform,
    yeojohnson_transform,
)


class DataValidationResult:
    def __init__(self, series, sample_size, dtype):
        self.series = series
        self.sample_size = sample_size
        self.dtype = dtype


class SelectionResult:
    def __init__(
        self,
        detector,
        series,
        reason,
        is_normal,
        transformation=None,
        skewness=None,
        kurtosis=None,
        validation=None,
        normality=None,
        steps=None,
    ):
        self.detector = detector
        self.series = series
        self.reason = reason
        self.is_normal = is_normal
        self.transformation = transformation
        self.skewness = skewness
        self.kurtosis = kurtosis
        self.validation = validation
        self.normality = normality
        self.steps = steps or []

    def report(self, anomalies=None):
        return format_selection_report(self, anomalies=anomalies)


def validate_series(series):
    data = pd.Series(series)

    if data.isnull().any():
        raise ValueError("Временной ряд содержит значения NaN")

    if not pd.api.types.is_numeric_dtype(data):
        raise TypeError("Временной ряд должен содержать числовые значения")

    data = data.astype(float)

    if len(data) < 3:
        raise ValueError("Размер выборки должен быть не меньше 3")

    return DataValidationResult(
        series=data,
        sample_size=len(data),
        dtype=str(data.dtype),
    )


def select_normal_detector(sample_size, alpha=0.05):
    if sample_size < 10:
        return DixonQDetector()
    if sample_size <= 30:
        return GrubbsDetector(alpha=alpha)
    if sample_size <= 100:
        return ChauvenetDetector()
    return ZScoreDetector()


def _try_normalizing_transformations(series, alpha, steps=None):
    transformations = (
        log_transform,
        boxcox_transform,
        yeojohnson_transform,
    )

    for transform in transformations:
        try:
            result = transform(series)
        except ValueError:
            if steps is not None:
                steps.append(
                    "Преобразование {0}: невозможно применить".format(
                        transform.__name__
                    )
                )
            continue

        normality = test_normality(result.series, alpha=alpha)
        if steps is not None:
            steps.append(
                "Преобразование {0}: p-value={1:.4f}, нормальность={2}".format(
                    result.method,
                    normality.p_value,
                    "да" if normality.is_normal else "нет",
                )
            )
        if normality.is_normal:
            return result

    return None


def select_detector_result(series, alpha=0.05):
    steps = []
    validation = validate_series(series)
    data = validation.series
    sample_size = validation.sample_size
    steps.append(
        "Проверка данных: NaN нет, тип={0}, размер выборки={1}".format(
            validation.dtype,
            sample_size,
        )
    )

    normality = test_normality(data, alpha=alpha)
    steps.append(
        "Проверка нормальности: метод={0}, p-value={1:.4f}, нормальность={2}".format(
            normality.method,
            normality.p_value,
            "да" if normality.is_normal else "нет",
        )
    )

    if normality.is_normal:
        detector = select_normal_detector(sample_size, alpha=alpha)
        steps.append(
            "Выбран критерий для нормального распределения: {0}".format(
                detector.__class__.__name__
            )
        )
        return SelectionResult(
            detector=detector,
            series=data,
            reason="normal_distribution",
            is_normal=True,
            validation=validation,
            normality=normality,
            steps=steps,
        )

    transformation = _try_normalizing_transformations(data, alpha, steps=steps)
    if transformation is not None:
        detector = select_normal_detector(sample_size, alpha=alpha)
        steps.append(
            "Выбрана нормализация: {0}".format(transformation.method)
        )
        steps.append(
            "После нормализации выбран критерий: {0}".format(
                detector.__class__.__name__
            )
        )
        return SelectionResult(
            detector=detector,
            series=transformation.series,
            reason="normal_after_transformation",
            is_normal=True,
            transformation=transformation,
            validation=validation,
            normality=normality,
            steps=steps,
        )



    skewness = float(stats.skew(data, bias=False))
    kurtosis = float(stats.kurtosis(data, fisher=True, bias=False))
    steps.append("Нормализация не выбрана: распределение осталось ненормальным")
    steps.append(
        "Дополнительные статистики: skewness={0:.4f}, kurtosis={1:.4f}".format(
            skewness,
            kurtosis,
        )
    )

    if not np.isfinite(skewness) or not np.isfinite(kurtosis):
        steps.append("Выбран критерий для сложного случая: MonteCarloDetector")
        return SelectionResult(
            detector=MonteCarloDetector(),
            series=data,
            reason="non_normal_complex_case",
            is_normal=False,
            skewness=skewness,
            kurtosis=kurtosis,
            validation=validation,
            normality=normality,
            steps=steps,
        )

    abs_skewness = abs(skewness)

    if kurtosis > 3:
        steps.append("Выбран критерий для тяжелых хвостов: ModifiedZScoreDetector")
        return SelectionResult(
            detector=ModifiedZScoreDetector(),
            series=data,
            reason="non_normal_heavy_tails",
            is_normal=False,
            skewness=skewness,
            kurtosis=kurtosis,
            validation=validation,
            normality=normality,
            steps=steps,
        )

    if abs_skewness <= 1:
        steps.append("Выбран критерий для умеренной асимметрии: IQRDetector")
        return SelectionResult(
            detector=IQRDetector(),
            series=data,
            reason="non_normal_moderate_skewness",
            is_normal=False,
            skewness=skewness,
            kurtosis=kurtosis,
            validation=validation,
            normality=normality,
            steps=steps,
        )

    if abs_skewness > 1:
        steps.append("Выбран критерий для сильной асимметрии: HampelDetector")
        return SelectionResult(
            detector=HampelDetector(),
            series=data,
            reason="non_normal_strong_skewness",
            is_normal=False,
            skewness=skewness,
            kurtosis=kurtosis,
            validation=validation,
            normality=normality,
            steps=steps,
        )

    steps.append("Выбран критерий для сложного случая: MonteCarloDetector")
    return SelectionResult(
        detector=MonteCarloDetector(),
        series=data,
        reason="non_normal_complex_case",
        is_normal=False,
        skewness=skewness,
        kurtosis=kurtosis,
        validation=validation,
        normality=normality,
        steps=steps,
    )


def select_detector(series, alpha=0.05):
    return select_detector_result(series, alpha=alpha).detector


def format_selection_report(selection, anomalies=None):
    lines = [
        "Автоматический выбор критерия",
        "------------------------------",
    ]

    for index, step in enumerate(selection.steps, start=1):
        lines.append("{0}. {1}".format(index, step))

    lines.extend(
        [
            "",
            "Итог:",
            "Критерий: {0}".format(selection.detector.__class__.__name__),
            "Причина выбора: {0}".format(selection.reason),
            "Нормализация: {0}".format(
                selection.transformation.method
                if selection.transformation is not None
                else "не применялась"
            ),
        ]
    )

    if anomalies is not None:
        anomaly_series = pd.Series(anomalies).astype(bool)
        anomaly_indexes = anomaly_series[anomaly_series].index.tolist()
        lines.append("Количество аномалий: {0}".format(len(anomaly_indexes)))
        lines.append("Индексы аномалий: {0}".format(anomaly_indexes))
    return "\n".join(lines)
