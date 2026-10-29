# Interface reasoning

The idea is to present a wrapper for a scikit-learn model, so that it can be immediately dropped into existing workflows. The wrapper extends the functionality by adding `calibrate()` and `predict_with_interval()` functions. The first of these calibrates the model, and the second returns the prediction interval at the requested percentile. The examples here use a fresh model and fit to some data, but a pre-fitted model could also be used (with a small tweak to the CLI option).

This is probably ideal for the scientist user, who might not have to do anything to their existing code other than initialise the UQModel class, and later calibrate and then get prediction intervals. This is, however, a bit awkward for the workflow calling it. We have to build a load of extra code around the class that handles things like model selection, data loading, fitting etc. This code could be quite brittle, and it would be impossible to account for every option that a calling program would want. Thus it could easily end up ballooning and size and complexity, without delivering any real value.

# TODOs

- Add type hints to the Python code.
- Add docstrings to the Python code.
- Extend the scikit-learn classes `BaseEstimator` and `RegressorMixin`. Then I might be able to simplify the class by removing the `fit()` and `predict()` 
functions.
- Build a better interface for the CLI interface. E.g.
  - Include loading a pre-fitted model from file.
  - Using a TOML (or other format) config file.
- Add some useful output, optionally logged to file. E.g.
  - Progress bar/info
  - Exceptions and warnings
  - Fitting/calibration statistics
- Make the class extensible for other UQ methods.
- Add tests (unit, component, end to end, etc).
- Build a better workflow example (e.g. use a real workflow engine, not just a simple bash script)
- Add a CI on GitHub to run tests and do linting (triggered by a pull request?).
- Add the option to pass in arrays of percentiles?
- Add the option to return the prediction along with the interval?
- Add utilities, e.g. for plotting.

# AI use

Just the Google summary, and occasionally continuing with the chat option.

If I had convenient access built into VSCode I would have used AI to quickly generate things like doc strings, type hints, documentation, etc.
