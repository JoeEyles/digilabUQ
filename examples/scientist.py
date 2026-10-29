from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from uq import UQModel


def load_data():
    return make_regression(
        n_samples=100, n_features=4, noise=10, random_state=42
    )


def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_calibrate, X_test, y_calibrate, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    return X_train, X_test, X_calibrate, y_train, y_test, y_calibrate


def main():
    X, y = load_data()
    X_train, X_test, X_calibrate, y_train, y_test, y_calibrate = split_data(X, y)
    uq_model = UQModel(MLPRegressor(hidden_layer_sizes=(8,), random_state=42))
    uq_model.fit(X_train, y_train)
    uq_model.calibrate(X_calibrate, y_calibrate)
    prediction_intervals = uq_model.predict_in_interval(X_test, 90)
    for y_val, interval in zip(y_test, prediction_intervals):
        print(f"Predicted {y_val} with 90% coverage: [{interval[0]}, {interval[1]}]")


if __name__ == "__main__":
    main()
