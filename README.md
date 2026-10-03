# Fourier Fingerprints of Ansaetze in Quantum Machine Learning

Experiments on the Fourier coefficient correlation (FCC) of variational
ansaetze: their fingerprints and their relation to training error on Fourier
series and a high energy physics regression task.
Models are built with [qml-essentials](https://github.com/cirKITers/qml-essentials)
on the [jaqsi](https://github.com/cirKITers/jaqsi) simulator, experiments run as
versioned [Fluksio](https://docs.fluksio.com) runs.

## Layout

- `fourier_fingerprints/`: the library (models and custom ansaetze, FCC and
  expressibility, datasets, training loops) and `pipeline.py`, the Fluksio flows
  fingerprint, surrogate, expressibility, train and encoding
- `dev/`: one folder per study with its driver (`run.py`), export (`export.py`),
  figures (`figures.py`), README, and reference CSVs;
  `dev/serve.sh` starts the engine
  - `s1-fingerprints`: FCC, expressibility and the random-coefficient surrogate
  - `s2-fourier-series`: training on 1D and 2D Fourier series
  - `s3-hep`: training on the $pp \to Z \to$ jets dataset, QFM and MLP
  - `s4-encodings`: FCC and training error of the encoding strategies
- `tests/`: unit tests, `uv run pytest`
- `data/`: the HEP datasets (not tracked)

## Getting started

```sh
uv sync                                               # Python 3.12 environment
RUNS=4 DEVICES=4 dev/serve.sh                         # the engine, store in ./.fluksio
uv run fluksio sync fourier_fingerprints/pipeline.py  # upload the flows
uv run fluksio run fingerprint --no-sync --defaults --wait  # one run
```

Each study README lists its cells and how to run, export and plot it. The drivers
submit to the running engine and skip cells that already ran, so an interrupted
grid resumes on the next call. Restart the engine after `uv sync`, and keep
`RUNS` $\times$ `DEVICES` near the number of cores.
