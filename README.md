# Relation Extraction

## Prerequisites

* Python 3.5.2+

* (Optional, but highly recommended) Create and use a virtual environment for isolating
  from system site directories - there are many options, but the most recommended is `venv`
  by the time of writing this. Creating virtual environment with `venv`: `python3 -m venv <path_to_your_virtual_env>`.
  Activating it is done with `source <path_to_your_virtual_env>/bin/activate`.

* To install the required packages, activate the virtual environment
  and run `pip3 install -r requirements.txt` or if you are without a virtual environment,
  run `pip3 install --user -r requirements.txt` to install them in a Python user
  install directory instead of a system directory.

## Structure

`data/`: Contains data sets for training, testing, validation and external resources.

`documentation/`: Documentation of the project.

`main.py`: The main configurable pipeline for training, testing and running models on data sets.

`models/`: Contains all our models which have a `fit()` and `predict()` method.

`notebooks/`: Contains Jupyter notebooks.

`predictions/`: Contains predictions.

`README.md`: Documentation of the repository.

`requirements.txt`: List of Python pip dependencies.

`results/`: Contains JSON files with evaluation results of the trained models and their parameters.

`transformers/`: Contains all feature transformers.

`utils.py`: A set of utility methods used for model evaluation, and others.
