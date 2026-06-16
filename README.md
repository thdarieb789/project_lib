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

## Используемая литература

1. Тюрин Ю. Н., Макаров А. А., Высоцкий И. Р., Ященко И. В. Теория вероятностей и статистика. - М.: МЦНМО, 2008.
2. Высоцкий И. Р., Ященко И. В. Теория вероятностей и статистика. 10-11 классы. - М.: Просвещение.
3. Мордкович А. Г., Семёнов П. В. Алгебра и начала математического анализа. 10 класс (элементы статистики). - М.: Мнемозина.
4. Прохоренок Н. А., Дронов В. А. Python 3. Самое необходимое. - СПб.: БХВ-Петербург.
5. Лутц М. Изучаем Python. - М.: Вильямс.
6. Документация Python. - URL: https://docs.python.org/3/
7. Документация библиотек NumPy, pandas, SciPy и Matplotlib (официальные сайты).
8. Учебные материалы образовательной платформы «Stepik» по программированию на Python.

## Лицензия

MIT
