# Spectral Ansatzes for Quantum Machine Learning (SAQML)


## Getting Started :rocket:

This repository uses [Kedro](https://kedro.org/). To get started, follow these steps:
1. Clone this repository
2. Install the pinned environment with [uv](https://docs.astral.sh/uv/): `uv sync`
3. Run the experiment: `uv run kedro run`

Experiments are automatically recorded using [MlFlow](https://mlflow.org/). You can view the experiments by
1. Running `uv run kedro mlflow ui`
2. Navigating to [http://127.0.0.1:5000](http://127.0.0.1:5000)

To visualize the nodes and pipeline
1. Run `uv run kedro viz`
2. Navigating to [http://127.0.0.1:4141](http://127.0.0.1:4141)

## Tweaking :wrench:

- To specify a pipeline: `kedro run --pipeline NAME` (see `src/saqml/pipeline_registry.py`)
- Parameters can be adjusted in `conf/base/parameters.yml` or as command line arguments `--params=<key1>=<value1>`
- Circuit diagrams in `docs/` are generated with `kedro run --pipeline visualize --params=model.circuit_type=<circuit>,model.n_qubits=4,model.draw=True`
- `slurm_job.sh` and the `sweep_*.sh` scripts are the SLURM job scripts used for the parameter sweeps