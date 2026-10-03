# Fourier Fingerprints of Ansatzes in QML


## Getting Started :rocket:

This repository uses [Kedro](https://kedro.org/). To get started, follow these steps:
1. Clone this repository
2. Install the pinned environment (Python 3.11) with [Poetry](https://python-poetry.org/): `poetry env use python3.11 && poetry install`
3. Run the experiment: `poetry run kedro run`

Experiments are automatically recorded using [MlFlow](https://mlflow.org/). You can view the experiments by
1. Running `poetry run kedro mlflow ui`
2. Navigating to [http://127.0.0.1:5000](http://127.0.0.1:5000)

To visualize the nodes and pipeline
1. Run `poetry run kedro viz`
2. Navigating to [http://127.0.0.1:4141](http://127.0.0.1:4141)

## Tweaking :wrench:

- To specify a pipeline: `kedro run --pipeline NAME` (see `src/fourier_fingerprints/pipeline_registry.py`)
- Parameters can be adjusted in `conf/base/parameters.yml` or as command line arguments `--params=<key1>=<value1>`
- Circuit diagrams in `docs/` are generated with `kedro run --pipeline visualize --params=model.circuit_type=<circuit>,model.n_qubits=4,model.draw=True`
- `slurm_job.sh` and the `sweep_*.sh` scripts are the SLURM job scripts used for the parameter sweeps. `slurm_job.sh` expects the repo at `~/fourier_fingerprints` and the environment in `.venv` (run `poetry config virtualenvs.in-project true` before `poetry install`)