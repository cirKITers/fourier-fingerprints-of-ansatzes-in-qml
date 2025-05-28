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

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def save_fig(fig, name, run_ids, experiment_id, font_size=16, scale=1):
    hs = generate_hash(run_ids)
    path = f"results/{experiment_id}/{hs}/"
    os.makedirs(path, exist_ok=True)
    print(f"Saving figure to {path}{name}.pdf")
    fig.update_layout(font=dict(size=font_size))
    fig.write_image(f"{path}{name}.pdf", scale=scale)


def get_run_ids(experiment_id):
    print(f"Searching experiment with id {experiment_id}")
    df = mlflow.search_runs([experiment_id])
    print(f"Found {len(df)} runs")
    if len(df[(df.status != "FINISHED")]) > 0:
        print(f"{df[(df.status != 'FINISHED')]} runs not finished")
    else:
        print("All runs finished")
    return df.run_id.to_list()


def cache_df(run_ids, df=None):
    # calculate hash
    hs = generate_hash(run_ids)

    # save df to cache
    path = f".cache/{hs}/"
    os.makedirs(path, exist_ok=True)

    if os.path.exists(f"{path}df.csv"):
        print(f"DF already exists: {hs}")
        df = pd.read_csv(f"{path}df.csv")
    else:
        if df is None:
            return None
        df.to_csv(f"{path}df.csv")
        print(f"Created DF cache: {hs}")
        df = pd.read_csv(f"{path}df.csv")

    return df


def get_color_iterator(option=0):
    if option == 0:
        main_colors_it = iter(plotly.colors.qualitative.Dark2)
        sec_colors_it = iter(plotly.colors.qualitative.Pastel2)
    elif option == 1:
        main_colors_it = iter(plotly.colors.qualitative.Dark2)
        sec_colors_it = iter(plotly.colors.qualitative.Pastel1)
    elif option == 2:
        main_colors_it = iter(plotly.colors.qualitative.Set1)
        sec_colors_it = iter(plotly.colors.qualitative.Pastel1)

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


def get_training_df(run_ids, cutoff_mse=-1, cutoff_steps=-1, metric="mse"):
    df = pd.DataFrame(
        columns=[
            "training_run_id",
            "ansatz",
            "qubits",
            "seed",
            f"{metric}",
            f"{metric}_min",
            f"{metric}_max",
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

        df.loc[it, "training_run_id"] = run_id
        df.loc[it, "ansatz"] = client.get_run(run_id).data.params["model.circuit_type"]
        df.loc[it, "qubits"] = int(client.get_run(run_id).data.params["model.n_qubits"])
        df.loc[it, "seed"] = int(client.get_run(run_id).data.params["seed"])
        steps = int(client.get_run(run_id).data.params["training.steps"])

        mse_hist = client.get_metric_history(run_id, f"{metric}")
        mse_values = np.empty((steps))
        mse_values[:] = np.nan

        mse_values[: len(mse_hist)] = [
            entity.value if entity.value > cutoff_mse else np.nan for entity in mse_hist
        ]

        df.loc[it, f"{metric}"] = mse_values
        df.loc[it, f"{metric}_min"] = np.min(mse_values[: len(mse_hist)])
        df.loc[it, f"{metric}_max"] = np.max(mse_values[: len(mse_hist)])
        df.loc[it, "steps"] = mse_values[: len(mse_hist)][
            mse_values > cutoff_steps
        ].size

    return df


def get_coefficient_df(run_ids, expr=False):
    if expr:
        df = pd.DataFrame(
            columns=[
                "coeff_run_id",
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
                "coeff_run_id",
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

        df.loc[it, "coeff_run_id"] = run_id
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
            "expr_run_id",
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

        df.loc[it, "expr_run_id"] = run_id
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


def visualize_boxplot(
    df,
    metric,
):
    qubit = df["qubits"].unique()[0]

    fig = make_subplots()
    main_colors_it, _ = get_color_iterator()
    fig.add_trace(
        go.Box(
            x=df.ansatz,
            y=df[metric],
            name=f"Metric: {metric.replace('_', ' ')}",
            marker=dict(color=rgb_to_rgba(next(main_colors_it), 0.5)),
            yaxis="y3",
            offsetgroup="Metric",
        ),
    )
    fig.add_trace(
        go.Box(
            x=df.ansatz,
            y=df.kl_divergence,
            name=f"KL Divergence",
            marker=dict(color=rgb_to_rgba(next(main_colors_it), 0.5)),
            yaxis="y2",
            offsetgroup="KL Divergence",
        ),
    )

    fig.add_trace(
        go.Box(
            x=df.ansatz,
            y=df.corr_mean,
            name=f"FCC",
            marker=dict(color=rgb_to_rgba(next(main_colors_it), 0.5)),
            yaxis="y",
            offsetgroup="FCC",
        ),
    )

    fig.add_trace(
        go.Box(
            x=df.ansatz,
            y=df.corr_w_mean,
            name=f"FCC Weighted",
            marker=dict(color=rgb_to_rgba(next(main_colors_it), 0.5)),
            yaxis="y",
            offsetgroup="FCC Weighted",
        ),
    )

    fig.update_yaxes(title_text=f"Correlation", secondary_y=False)
    fig.update_yaxes(title_text="KL Divergence", secondary_y=True)
    fig.update_layout(
        title=f"FCC and Expressibility ({qubit} Qubits, Metric: {metric})",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="FCC"),  # yaxis = "y", attached to x-axis
        yaxis2=dict(
            position=0,
            title="KL Divergence",  # yaxis = "y2", pos 0, free from x-axis
            side="left",
            anchor="free",
            overlaying="y",
        ),
        yaxis3=dict(
            title="Metric",
            side="right",
            anchor="x",  # yaxis = "y3", attached to x-axis
            overlaying="y",
        ),
        xaxis=dict(domain=[0.15, 0.9], tickangle=20),
        boxmode="group",
        margin=dict(l=50, r=0, b=30, t=100),
    )

    return fig


def visualize_scatter(df, ansatz_ids, metric, weighted=True):
    corr_mean = "corr_mean" if not weighted else "corr_w_mean"

    fig = go.Figure()
    symbols = get_symbol_iterator()
    main_colors_it, _ = get_color_iterator()
    symbol = next(symbols)
    for ansatz_id in ansatz_ids:
        _df = df[(df.ansatz_id == ansatz_id)]
        if len(_df) == 0:
            print(f"No data for ansatz_id={ansatz_id}")
            continue
        ansatz = _df["ansatz"].unique()[0]
        fig.add_scatter(
            x=[_df[corr_mean].mean()],
            y=[_df[metric].mean()],
            error_x=dict(
                type="data",
                array=[_df[corr_mean].std()],
                visible=True,
            ),
            error_y=dict(
                type="data",
                array=[_df[metric].std()],
                visible=True,
            ),
            mode="markers",
            name=f"{ansatz}",
            marker=dict(color=next(main_colors_it), symbol=symbol),
        )

    fig.update_layout(
        title_text="Direct Correlation ({qubit} Qubits, Metric: {metric})",
        template="plotly_white",
        xaxis=dict(
            title=("Correlation Mean" if not weighted else "Weighted Correlation Mean"),
        ),
        yaxis=dict(
            title=metric.title(),
            showgrid=False,
        ),
        showlegend=True,
    )

    return fig


def visualize_expr_scatter(df, ansatz_ids, metric, weighted=False):
    corr_mean = "corr_mean" if not weighted else "corr_w_mean"

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    main_colors_it, sec_colors_it = get_color_iterator(option=2)
    for ansatz_id in ansatz_ids:
        _df = df[(df.ansatz_id == ansatz_id)]
        if len(_df) == 0:
            print(f"No data for ansatz_id={ansatz_id}")
            continue
        ansatz = _df["ansatz"].unique()[0]
        fig.add_scatter(
            x=[_df[corr_mean].mean()],
            y=[_df[metric].mean()],
            error_x=dict(
                type="data",
                array=[_df[metric].std()],
                visible=True,
            ),
            error_y=dict(
                type="data",
                array=[_df[corr_mean].std()],
                visible=True,
            ),
            mode="markers",
            name=f"{ansatz}",
            marker=dict(color=next(main_colors_it), symbol="circle", size=15),
            secondary_y=False,
            showlegend=True,
        )

        fig.add_scatter(
            x=[_df["kl_divergence"].mean()],
            y=[_df[metric].mean()],
            error_x=dict(
                type="data",
                array=[_df[metric].std()],
                visible=True,
            ),
            error_y=dict(
                type="data",
                array=[_df["kl_divergence"].std()],
                visible=True,
            ),
            mode="markers",
            name=f"{ansatz} (EXPR)",
            marker=dict(color=next(sec_colors_it), symbol="diamond", size=10),
            secondary_y=True,
            showlegend=False,
        )

    fig.update_layout(
        title_text="Direct Correlation ({qubit} Qubits, Metric: {metric})",
        template="plotly_white",
        xaxis=dict(
            title=metric.title(),
            showgrid=False,
        ),
        yaxis=dict(
            title=("Correlation Mean" if not weighted else "Weighted Correlation Mean"),
        ),
        yaxis2=dict(
            title=("Expressibility"),
            side="right",
        ),
    )

    return fig


def visualize_heatmap(df, selected_seed, weighted):
    ansaetze = df.ansatz.unique()
    qubit = df["qubits"].unique()[0]

    fig = make_subplots(rows=1, cols=len(ansaetze), subplot_titles=ansaetze)

    for it, ansatz in enumerate(ansaetze):
        _df = df[(df.ansatz == ansatz) & (df.seed == selected_seed)]
        if len(_df) == 0:
            print(f"No data for q={qubit}, ansatz={ansatz}, seed={selected_seed}")
            continue
        sub_fig_trace = get_plotly_artifact(
            _df.coeff_run_id.item(),
            f"coefficients_correlated_{'w' if weighted else 'uw'}",
        )

        fig.add_trace(sub_fig_trace, row=1, col=it + 1)
        fig.update_xaxes(dict(title="Coefficients"), row=1, col=it + 1)
        fig.update_yaxes(
            dict(
                title="Coefficients" if it == 0 else "",
                autorange="reversed",
                scaleanchor="x",
            ),
            row=1,
            col=it + 1,
        )

    fig.update_layout(
        title_text=(
            f"{'Weighted ' if weighted else ''}Correlation of Coefficients for Different Ansaetze ({qubit} Qubits)"
        ),
        template="plotly_white",
        height=400,
        width=300 * it,
        coloraxis={"colorscale": "Sunset"},
    )
