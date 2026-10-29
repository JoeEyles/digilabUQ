# digilabUQ

## Setup

Clone the repo with
```sh
git clone git@github.com:JoeEyles/digilabUQ.git
```

Setup a conda environment (optionally use `mamba` for speed)
```sh
conda create -n digilab
```

Install with pip (from the root of the repo)
```sh
pip install .
```

## Usage

Use the `UQModel` class as if it were a scikit-learn model with an additional `predict_with_interval()` function. This function takes input data `X` and a target coverage percentile and returns intervals as a list of lists. For example:

```python
    # Import the uncertainty quantification class
    from uq import UQModel
    # Initialise the UQModel class with the scikit-learn model to be used for fit() and predict()
    uq_model = UQModel(MLPRegressor(hidden_layer_sizes=(8,), random_state=42))
    # Fit the model as usual. This is called on the input scikit-learn model
    uq_model.fit(X_train, y_train)
    # Calibrate the model
    uq_model.calibrate(X_calibrate, y_calibrate)
    # Compute prediction intervals at 90% for each input sample
    prediction_intervals = uq_model.predict_with_interval(X_test, 90)
    # Print the result: an interval for each target value 
    for y_val, interval in zip(y_test, prediction_intervals):
        print(f"Prediction for {y_val} with 90% coverage: [{interval[0]}, {interval[1]}]")
```

See the `examples` folder for a full runnable example.
