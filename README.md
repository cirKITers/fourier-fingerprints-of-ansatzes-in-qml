# Spectral Ansatzes for Quantum Machine Learning (SAQML)


## Getting Started :rocket:

This repository uses [Kedro](https://kedro.org/). To get started, follow these steps:
1. Clone this repository
2. Run `poetry install` or `pip install -r requirements.in`
3. Run the experiment: `kedro run`

Experiments are automatically recorded using [MlFlow](https://mlflow.org/). You can view the experiments by
1. Running `poetry run kedro mlflow ui`
2. Navigating to [http://127.0.0.1:5000](http://127.0.0.1:5000)

To visualize the nodes and pipeline
1. Run `poetry run kedro viz`
2. 2. Navigating to [http://127.0.0.1:4141](http://127.0.0.1:4141)

## Tweaking :wrench:

- To specify a pipeline: `kedro run --pipeline NAME`
 - Parameters can be adjusted in `conf\base\parameters.yml` or as command line arguments `--params=<key1>=<value1>`