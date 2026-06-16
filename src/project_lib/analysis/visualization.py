import pandas as pd


def plot_anomalies(
    series,
    anomalies=None,
    detector=None,
    ax=None,
    title="Anomaly detection",
    line_label="Series",
    anomaly_label="Anomalies",
    show=False,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Для визуализации установите matplotlib"
        ) from exc

    data = pd.Series(series)

    if anomalies is None:
        if detector is None:
            anomalies = pd.Series(False, index=data.index)
        elif hasattr(detector, "detector") and hasattr(detector, "series"):
            anomalies = detector.detector.detect(detector.series)
        else:
            anomalies = detector.detect(data)
    else:
        anomalies = pd.Series(anomalies, index=data.index).astype(bool)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.plot(data.index, data.values, label=line_label, color="#1f77b4")

    anomaly_points = data[anomalies]
    if not anomaly_points.empty:
        ax.scatter(
            anomaly_points.index,
            anomaly_points.values,
            label=anomaly_label,
            color="#d62728",
            zorder=3,
        )

    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if show:
        plt.show()

    return ax
