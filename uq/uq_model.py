import numpy as np
from sklearn.exceptions import NotFittedError


class NotCalibratedError(ValueError, AttributeError):
    pass


class UQModel:
    def __init__(self, model):
        self._model = model
        self._fitted = False
        self._calibrated = False

    def fit(self, X, y):
        self._model.fit(X, y)
        self._fitted = True

    def calibrate(self, X, y):
        if not self._fitted:
            raise NotFittedError()
        self._calibration_errors = np.abs(self._model.predict(X) - y)
        self._calibration_errors = np.sort(self._calibration_errors, descending=False)
        self._calibrated = True

    def get_uncertainty_bound(self, percentile):
        if not self._calibrated:
            raise NotCalibratedError()
        if percentile < 0 or percentile > 100:
            raise ValueError("Percentile must be between 0 and 100.")
        alpha = 1 - percentile/100
        n = len(self._calibration_errors)
        k = np.ceil((n+1)*(1-alpha)) - 1  # minus 1 for 0-indexed array
        return self._calibration_errors[int(k)]

    def predict(self, X):
        if not self._fitted:
            raise NotFittedError()
        return self._model.predict(X)

    def predict_to_uncertainty_interval(self, X, percentile):
        if not self._fitted:
            raise NotFittedError()
        if not self._calibrated:
            raise NotCalibratedError()
        prediction = self.predict(X)
        bound = self.get_uncertainty_bound(percentile)
        return [prediction - bound, prediction + bound]
