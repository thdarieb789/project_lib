from project_lib import TimeSeries, IQRDetector

ts = TimeSeries([10, 12, 11, 13, 999])
detector = IQRDetector()

print(ts.find_anomalies(detector))