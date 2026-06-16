# project-lib

Статистическое обнаружение выбросов и аномалий во временных рядах.

`project-lib` предоставляет набор классических методов поиска выбросов
с единым интерфейсом, а также вспомогательные средства для проверки
нормальности, преобразования данных и автоматического подбора детектора.

**Проект на PyPI:** https://pypi.org/project/anomaly-detector-lib/

## Установка

```bash
pip install anomaly-detector-lib
```

Имя для импорта в коде - `project_lib`.

## Детекторы

- `ChauvenetDetector` - критерий Шовене
- `DixonQDetector` - Q-критерий Диксона
- `GrubbsDetector` - критерий Граббса
- `HampelDetector` - фильтр Хампеля
- `IQRDetector` - правило межквартильного размаха
- `ModifiedZScoreDetector` - модифицированный z-score (на основе MAD)
- `MonteCarloDetector` - обнаружение методом Монте-Карло
- `ZScoreDetector` - классический z-score

## Быстрый старт

```python
import pandas as pd
from project_lib import TimeSeries, ZScoreDetector

series = pd.Series([1, 2, 1, 2, 100, 2, 1])
detector = ZScoreDetector()
anomalies = detector.detect(series)
print(anomalies)
```

## Анализ данных

Проверка нормальности (`test_normality`), преобразования (`log_transform`,
`sqrt_transform`, `boxcox_transform`, `yeojohnson_transform`), автоматический
подбор детектора (`select_detector`, `select_normal_detector`) и визуализация
(`plot_anomalies`).

