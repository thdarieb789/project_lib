from project_lib import TimeSeries, GrubbsDetector
from project_lib.analysis.visualization import (plot_anomalies)

ts = TimeSeries([10, 12, 11, 13, 45]) 
detector = GrubbsDetector()

print(ts.find_anomalies_auto())
plot_anomalies([10, 12, 11, 13, 45], detector=detector, show=True)