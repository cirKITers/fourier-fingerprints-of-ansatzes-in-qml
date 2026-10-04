import inspect

import numpy as np

from fourier_fingerprints import pipeline
from fourier_fingerprints.data import fourier_series, hep_dataset
from fourier_fingerprints.metrics import correlation_stats, expressibility, fcc
from fourier_fingerprints.model import create_model
from fourier_fingerprints.train import train_fourier_series, train_mlp, train_qfm


def defaults(fn):
    return {
        name: p.default
        for name, p in inspect.signature(fn).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


def test_flow_inputs_follow_the_library_defaults():
    # the train flow defaults to the QFM on the Fourier series, so train_qfm wins
    library = {
        **defaults(create_model),
        **defaults(train_mlp),
        **defaults(train_qfm),
        "n_events": defaults(hep_dataset)["n_events"],
        "tol": defaults(correlation_stats)["tol"],
    }
    encoding = {
        **defaults(create_model),
        **defaults(fcc),
        **defaults(fourier_series),
        **defaults(train_fourier_series),
    }
    flows = (
        (pipeline.fingerprint_flow, library),
        (pipeline.surrogate_flow, library),
        (pipeline.train_flow, library),
        (pipeline.encoding_flow, encoding),
    )
    for flow, values in flows:
        for port in flow.inputs:
            if port.name in values:
                value = values[port.name]
                value = list(value) if isinstance(value, tuple) else value
                assert port.initial == value, (flow.name, port.name)

    ports = {p.name: p.initial for p in pipeline.expressibility_flow.inputs}
    assert ports["n_samples"] == defaults(expressibility)["n_samples"]
    assert ports["n_bins"] == defaults(expressibility)["n_bins"]


def test_jsonable_drops_non_finite_values():
    # Fluksio refuses NaN and infinity on every port
    value = {"a": np.array([[1.0, np.nan]]), "b": np.float64(np.inf), "c": np.int64(2)}
    assert pipeline.jsonable(value) == {"a": [[1.0, None]], "b": None, "c": 2}


def test_fourier_series_target_power():
    model = create_model(3, 1, "Circuit_9", encoding_strategy="binary")
    _, _, c = fourier_series(model, 1000, zero_centered=False, prune=True)
    _, y_p, c_p = fourier_series(
        model, 1000, zero_centered=False, prune=True, target_power=0.02
    )
    assert np.isclose(np.sum(np.abs(c_p / c_p.size) ** 2), 0.02, rtol=1e-12)
    assert np.isclose(np.mean(y_p**2), 0.02, rtol=1e-12)
    # rescaled on the pruned support, the series has the coefficients c / K
    assert np.allclose(c_p / np.linalg.norm(c_p), c / np.linalg.norm(c))
    assert (c != 0).sum() < c.size
    assert np.allclose(np.fft.fftshift(np.fft.fft(y_p)) / y_p.size, c_p / c_p.size)
