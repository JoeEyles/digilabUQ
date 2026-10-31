from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import argparse
import csv
import sys
import numpy as np
from uq_model import UQModel


def parse_args():
    parser = argparse.ArgumentParser(description="UQ CLI")
    parser.add_argument(
        "--coverage-percentile",
        help="Percentile between 0 and 100",
        type=float,
        default=90
    )
    parser.add_argument(
        "--input-data-path",
        help="TODO: Path to data csv?",
        type=str,
        default="examples/data.csv"
    )
    parser.add_argument(
        "--output-data-path",
        help="TODO: Path to data csv?",
        type=str,
        default="out.csv"
    )
    args = parser.parse_args()
    return args


def load_data(data_path):
    with open(data_path, "r", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        X = np.array([
            [float(row["x1"]), float(row["x2"]), float(row["x3"]), float(row["x4"])]
            for row in rows
        ])
        y = np.array([float(row["y"]) for row in rows])
    return X, y


def output_data(data_path, data_intervals):
    with open(data_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["lower_bound", "upper_bound"])
        for interval in data_intervals:
            writer.writerow([interval[0], interval[1]])


def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_calibrate, X_test, y_calibrate, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    return X_train, X_test, X_calibrate, y_train, y_test, y_calibrate


def fit_and_compute_intervals():
    args = parse_args()
    X, y = load_data(args.input_data_path)
    X_train, X_test, X_calibrate, y_train, y_test, y_calibrate = split_data(X, y)
    uq_model = UQModel(MLPRegressor(hidden_layer_sizes=(8,), random_state=42))
    uq_model.fit(X_train, y_train)
    uq_model.calibrate(X_calibrate, y_calibrate)
    prediction_intervals = uq_model.predict_with_interval(X_test, args.coverage_percentile)
    output_data(args.output_data_path, prediction_intervals)


def main():
    try:
        fit_and_compute_intervals()
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    main()  # TODO: debugging
