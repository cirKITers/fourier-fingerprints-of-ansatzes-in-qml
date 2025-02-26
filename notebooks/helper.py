import plotly
from plotly.validators.scatter.marker import SymbolValidator
import re
import json
import hashlib
import pandas as pd
import mlflow
import numpy as np
import os
from rich.progress import track


def save_fig(fig, name, run_ids, experiment_id, font_size=16, scale=1):
    hs = generate_hash(run_ids)
    path = f"results/{experiment_id}/{hs}/"
    os.makedirs(path, exist_ok=True)
    print(f"Saving figure to {path}{name}.pdf")
    fig.update_layout(font=dict(size=font_size))
    fig.write_image(f"{path}{name}.pdf", scale=scale)


def get_color_iterator():
    main_colors_it = iter(plotly.colors.qualitative.Dark2)
    sec_colors_it = iter(plotly.colors.qualitative.Pastel2)

    return main_colors_it, sec_colors_it


def get_symbol_iterator():
    raw_symbols = SymbolValidator().values
    symbols = []
    for i in range(0, len(raw_symbols), 12):
        symbols.append(raw_symbols[i])

    return iter(symbols)


def generate_hash(run_ids):
    hs = hashlib.md5(repr(run_ids).encode("utf-8")).hexdigest()
    return hs


def read_from_html(path):
    with open(path) as f:
        html = f.read()
    call_arg_str = re.findall(r"Plotly\.newPlot\((.*)\)", html[-(2**16) :])[0]
    call_args = json.loads(f"[{call_arg_str}]")
    plotly_json = {"data": call_args[1], "layout": call_args[2]}

    return plotly.io.from_json(json.dumps(plotly_json))


def get_plotly_artifact(run_id, identifier="coefficients_correlated"):
    client = mlflow.tracking.MlflowClient()

    fig_path = client.download_artifacts(run_id, f"{identifier}.html", "./")
    fig = read_from_html(fig_path)
    fig_trace = fig.data[0]
    fig_trace.update(
        # coloraxis=f"coloraxis",
        zmax=1.0,
        zmin=0.0,
    )

    os.remove(fig_path)

    return fig_trace


def rgb_to_rgba(rgb_value: str, alpha: float):
    """
    Adds the alpha channel to an RGB Value and returns it as an RGBA Value
    :param rgb_value: Input RGB Value
    :param alpha: Alpha Value to add  in range [0,1]
    :return: RGBA Value
    """
    return f"rgba{rgb_value[3:-1]}, {alpha})"


def get_training_df(run_ids, cutoff_mse=-1, cutoff_steps=-1):
    df = pd.DataFrame(
        columns=[
            "run_id",
            "ansatz",
            "qubits",
            "seed",
            "mse",
            "mse_min",
            "steps",
        ]
    )

    for it, run_id in track(
        enumerate(run_ids), description="Collecting training data..", total=len(run_ids)
    ):
        client = mlflow.tracking.MlflowClient()
        if client.get_run(run_id).info.status != "FINISHED":
            print(f"Run {run_id} not finished")
            continue

        df.loc[it, "run_id"] = run_id
        df.loc[it, "ansatz"] = client.get_run(run_id).data.params["model.circuit_type"]
        df.loc[it, "qubits"] = int(client.get_run(run_id).data.params["model.n_qubits"])
        df.loc[it, "seed"] = int(client.get_run(run_id).data.params["seed"])
        steps = int(client.get_run(run_id).data.params["training.steps"])

        mse_hist = client.get_metric_history(run_id, "mse")
        mse_values = np.empty((steps))
        mse_values[:] = np.nan

        mse_values[: len(mse_hist)] = [entity.value for entity in mse_hist]

        df.loc[it, "mse"] = mse_values[mse_values > cutoff_mse]
        df.loc[it, "mse_min"] = np.min(mse_values[: len(mse_hist)])
        df.loc[it, "steps"] = mse_values[: len(mse_hist)][
            mse_values > cutoff_steps
        ].size

    return df


def get_coefficient_df(run_ids, expr=False):
    if expr:
        df = pd.DataFrame(
            columns=[
                "run_id",
                "ansatz",
                "qubits",
                "layer_multiplier",
                "seed",
                "kl_divergence",
                "coefficients_correlation_mean",
                "coefficients_correlation_max",
                "coefficients_correlation_min",
                "coefficients_correlation_variance",
            ]
        )
    else:
        df = pd.DataFrame(
            columns=[
                "run_id",
                "ansatz",
                "qubits",
                "layer_multiplier",
                "seed",
                "coefficients_correlation_mean",
                "coefficients_correlation_weighted_mean",
                "coefficients_correlation_max",
                "coefficients_correlation_min",
                "coefficients_correlation_variance",
            ]
        )

    for it, run_id in track(
        enumerate(run_ids),
        description="Collecting coefficients data..",
        total=len(run_ids),
    ):
        client = mlflow.tracking.MlflowClient()
        if client.get_run(run_id).info.status != "FINISHED":
            print(f"Run {run_id} not finished")
            continue

        df.loc[it, "run_id"] = run_id
        df.loc[it, "ansatz"] = client.get_run(run_id).data.params["model.circuit_type"]
        df.loc[it, "qubits"] = int(client.get_run(run_id).data.params["model.n_qubits"])

        df.loc[it, "seed"] = int(client.get_run(run_id).data.params["seed"])

        df.loc[it, "coefficients_correlation_mean"] = client.get_run(
            run_id
        ).data.metrics["coefficients_correlation_mean"]

        if (
            "coefficients_correlation_weighted_mean"
            in client.get_run(run_id).data.metrics
        ):
            df.loc[it, "coefficients_correlation_weighted_mean"] = client.get_run(
                run_id
            ).data.metrics["coefficients_correlation_weighted_mean"]

        if "coefficients_correlation_max" in client.get_run(run_id).data.metrics:
            df.loc[it, "coefficients_correlation_max"] = client.get_run(
                run_id
            ).data.metrics["coefficients_correlation_max"]

        if "coefficients_correlation_min" in client.get_run(run_id).data.metrics:
            df.loc[it, "coefficients_correlation_min"] = client.get_run(
                run_id
            ).data.metrics["coefficients_correlation_min"]

        df.loc[it, "coefficients_correlation_variance"] = client.get_run(
            run_id
        ).data.metrics["coefficients_correlation_variance"]

        if "model.layer_multiplier" in client.get_run(run_id).data.params:
            df.loc[it, "layer_multiplier"] = int(
                client.get_run(run_id).data.params["model.layer_multiplier"]
            )
        if "expressibility" in client.get_run(run_id).data.metrics:
            df.loc[it, "kl_divergence"] = client.get_run(run_id).data.metrics[
                "expressibility"
            ]

    return df


def get_expressibility_df(run_ids):
    df = pd.DataFrame(
        columns=[
            "run_id",
            "ansatz",
            "qubits",
            "seed",
            "kl_divergence",
        ]
    )

    for it, run_id in track(
        enumerate(run_ids),
        description="Collecting expressibility data..",
        total=len(run_ids),
    ):
        client = mlflow.tracking.MlflowClient()
        if client.get_run(run_id).info.status != "FINISHED":
            print(f"Run {run_id} not finished")
            continue

        df.loc[it, "run_id"] = run_id
        df.loc[it, "ansatz"] = client.get_run(run_id).data.params["model.circuit_type"]
        df.loc[it, "qubits"] = int(client.get_run(run_id).data.params["model.n_qubits"])

        df.loc[it, "seed"] = int(client.get_run(run_id).data.params["seed"])

        df.loc[it, "kl_divergence"] = client.get_run(run_id).data.metrics[
            "expressibility"
        ]

    return df


def assign_ansatz_id(df):
    # add a column with name "ansatz_id" where each ansatz has a unique id
    df["ansatz_id"] = df["ansatz"].factorize()[0]
    return df
