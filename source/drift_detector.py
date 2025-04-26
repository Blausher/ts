class DriftDetector:
    def __init__(self, window_size=20, ewma_alpha=0.3, cusum_threshold=5):
        self.window_size = window_size
        self.ewma_alpha = ewma_alpha
        self.cusum_threshold = cusum_threshold
        self.ewma_prev = None
        self.cusum_pos = 0
        self.cusum_neg = 0

    def detect(self, series: pd.Series) -> bool:
        if len(series) < self.window_size:
            return False

        rolling_mean = series.rolling(window=self.window_size).mean()
        rolling_std = series.rolling(window=self.window_size).std()

        val = series.iloc[-1]
        mean = rolling_mean.iloc[-1]
        std = rolling_std.iloc[-1]

        # EWMA
        if self.ewma_prev is None:
            self.ewma_prev = val
        ewma = self.ewma_alpha * val + (1 - self.ewma_alpha) * self.ewma_prev
        self.ewma_prev = ewma

        ewma_deviation = abs(val - ewma)

        # CUSUM
        k = 0.5 * std
        self.cusum_pos = max(0, self.cusum_pos + val - mean - k)
        self.cusum_neg = min(0, self.cusum_neg + val - mean + k)

        cusum_alert = self.cusum_pos > self.cusum_threshold or abs(self.cusum_neg) > self.cusum_threshold

        # Логика триггера разладки
        if ewma_deviation > 2 * std or cusum_alert:
            print(f"Drift detected: EWMA deviation = {ewma_deviation}, CUSUM = {self.cusum_pos}/{self.cusum_neg}")
            return True

        return False