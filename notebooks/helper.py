import plotly
from plotly.validator_cache import ValidatorCache
import plotly.figure_factory as ff
import re
import json
import hashlib
import pandas as pd
import mlflow
import numpy as np
import os
from rich.progress import track
import math
import ast
import io

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# pio.kaleido.scope.mathjax = None


class design:
    marker_size = 18
    marker_line_width = 1
    marker_a_opacity = 1.0
    marker_b_opacity = 1.0
    marker_a_style = "cross"
    marker_a_color = "#009682"
    marker_b_style = "x"
    marker_b_color = "#DF9B1B"
    legend_color = "#002D4C"
    colorscale = "Sunset"
    annotation_font_offset = 0
    tick_font_offset = 1
    large_tick_font_offset = 0
    hm_tickangle = 0
    rel_tickangle = 30
    font_size = 22


def _coerce_export_number(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def save_fig(
    fig,
    name,
    run_ids,
    experiment_id,
    font_size=design.font_size,
    scale=1,
    showlegend=True,
    tight=False,
    large_ticks=False,
):
    hs = generate_hash(run_ids)
    path = f"results/{experiment_id}/{hs}/"
    os.makedirs(path, exist_ok=True)
    print(f"Saving figure to {path}{name}.pdf")
    fig.update_layout(
        font=dict(size=font_size),
        showlegend=showlegend,
        yaxis=dict(
            tickfont=dict(
                size=(
                    font_size - design.tick_font_offset
                    if not large_ticks
                    else font_size - design.large_tick_font_offset
                )
            )
        ),
        xaxis=dict(
            tickfont=dict(
                size=(
                    font_size - design.tick_font_offset
                    if not large_ticks
                    else font_size - design.large_tick_font_offset
                )
            )
        ),
    )
    fig.update_annotations(font_size=font_size - design.annotation_font_offset)
    if tight:
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
        )
    fig.write_image(
        f"{path}{name}.pdf",
        width=_coerce_export_number(fig.layout.width),
        height=_coerce_export_number(fig.layout.height),
        scale=_coerce_export_number(scale),
    )


def get_run_ids(experiment_id):
    if experiment_id is None:
        return None
    print(f"Searching experiment with id {experiment_id}")
    df = mlflow.search_runs([experiment_id])
    print(f"Found {len(df)} runs")
    if len(df[(df.status != "FINISHED")]) > 0:
        print(f"{df[(df.status != 'FINISHED')]} runs not finished")
    else:
        print("All runs finished")
    return df.run_id.to_list()


def tickval_to_tex(tickvals, use_latex=False, optimize=True):
    ticktext = []
    ct = -1
    for tick in tickvals:
        t = tick.replace("+", "").split("_")
        if len(t) == 2:
            if use_latex:
                ticktext.append(f"${t[0]}_{{{t[1]}}}$")
            else:
                ticktext.append(f"{t[1]}")
        elif len(t) == 3:
            if optimize:
                if int(t[1]) > ct:
                    if use_latex:
                        ticktext.append(f"${t[0]}_{{{t[1]},*}}$")
                    else:
                        ticktext.append(f"{t[1]},*")
                    ct = int(t[1])
                else:
                    ticktext.append("")
            else:
                ticktext.append(f"${t[0]}_{{{t[1]},{t[2]}}}$")

    return ticktext


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


def get_symbol_iterator(start=0):
    SymbolValidator = ValidatorCache.get_validator("scatter.marker", "symbol")
    raw_symbols = SymbolValidator.values
    symbols = []
    for i in range(start * 12, len(raw_symbols) - (start * 12), 12):
        symbols.append(raw_symbols[i])

    return iter(symbols)


def generate_hash(run_ids):
    hs = hashlib.md5(repr(run_ids).encode("utf-8")).hexdigest()
    return hs


# def read_from_html(path):
#     with open(path) as f:
#         html = f.read()
#     call_arg_str = re.findall(r"Plotly\.newPlot\((.*)\)", html[-(2**16) :])[0]
#     call_args = json.loads(f"[{call_arg_str}]")
#     plotly_json = {"data": call_args[1], "layout": call_args[2]}

#     return plotly.io.from_json(json.dumps(plotly_json), skip_invalid=True)


def read_from_html(path):
    with open(path) as f:
        html = f.read()
    call_arg_str = re.findall(r"Plotly\.newPlot\((.*)\)", html)[0]
    call_args = json.loads(f"[{call_arg_str}]")
    plotly_json = {"data": call_args[1], "layout": call_args[2]}

    return plotly.io.from_json(json.dumps(plotly_json), skip_invalid=True)


def get_plotly_heatmap(run_id, identifier="coefficients_correlated", automax=True):
    client = mlflow.tracking.MlflowClient()

    fig_path = client.download_artifacts(run_id, f"{identifier}.html", "./")
    fig = read_from_html(fig_path)
    data_z = np.abs(np.array(fig.data[0].z, dtype=np.float64))

    if automax:
        zmax = np.nanmax(data_z)
        if zmax > 0.1:
            zmax = math.ceil(zmax * 10) / 10
        elif zmax > 0.01:
            zmax = math.ceil(zmax * 100) / 100
        elif zmax > 0.001:
            zmax = math.ceil(zmax * 1000) / 1000
    else:
        zmax = 1.0
    zmin = 0.0

    fig_trace = go.Heatmap(
        z=data_z,
        y=fig.data[0].y,
        x=fig.data[0].x,
        hoverongaps=False,
        colorscale=design.colorscale,
        zmax=zmax,
        zmin=zmin,
        coloraxis=f"coloraxis",
    )

    os.remove(fig_path)

    return fig_trace


def beautify_circuit_name(circuit):

    if circuit.lower() == "hardware_efficient":
        circuit = "HEA"
    elif circuit.lower() == "circuit_yzy_entangling":
        circuit = "Circuit YZY Ent."

    circuit = circuit.replace("_", " ")
    circuit = circuit.replace("circuit", "C")
    circuit = circuit.replace("Circuit", "C")
    return circuit


def get_plotly_figure(run_id, identifier):
    client = mlflow.tracking.MlflowClient()
    try:
        fig_path = client.download_artifacts(run_id, f"{identifier}.html", "./")
    except FileNotFoundError:
        print(f"File {fig_path} not found for run id {run_id}")
        return None
    fig = read_from_html(fig_path)

    os.remove(fig_path)

    return fig


def get_csv_artifact(run_id, identifier):
    client = mlflow.tracking.MlflowClient()
    try:
        fig_path = client.download_artifacts(run_id, f"{identifier}.csv", "./")
    except FileNotFoundError:
        print(f"File {fig_path} not found for run id {run_id}")
        return None
    df = pd.read_csv(fig_path)

    os.remove(fig_path)

    return df


def rgb_to_rgba(rgb_value: str, alpha: float):
    """
    Adds the alpha channel to an RGB Value and returns it as an RGBA Value
    :param rgb_value: Input RGB Value
    :param alpha: Alpha Value to add  in range [0,1]
    :return: RGBA Value
    """
    return f"rgba{rgb_value[3:-1]}, {alpha})"


def get_training_df(
    run_ids,
    cutoff_mse=-1,
    cutoff_steps=-1,
    metrics=["mse"],
    run_id_tag="training_run_id",
):
    if run_ids is None:
        return None
    df = pd.DataFrame(
        columns=[
            run_id_tag,
            "ansatz",
            "qubits",
            "seed",
            *[f"{metric}" for metric in metrics],
            *[f"{metric}_min" for metric in metrics],
            *[f"{metric}_max" for metric in metrics],
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
        if run_id_tag != "classical_training_run_id":
            df.loc[it, "ansatz"] = client.get_run(run_id).data.params[
                "model.circuit_type"
            ]
            df.loc[it, "qubits"] = int(
                client.get_run(run_id).data.params["model.n_qubits"]
            )
        df.loc[it, "seed"] = int(client.get_run(run_id).data.params["seed"])
        steps = int(client.get_run(run_id).data.params["training.steps"])

        for metric in metrics:
            mse_hist = client.get_metric_history(run_id, f"{metric}")
            mse_values = np.empty((steps))
            mse_values[:] = np.nan

            mse_values[: len(mse_hist)] = [
                entity.value if entity.value > cutoff_mse else np.nan
                for entity in mse_hist
            ]

            df.loc[it, f"{metric}"] = mse_values
            df.loc[it, f"{metric}_min"] = np.min(mse_values[: len(mse_hist)])
            df.loc[it, f"{metric}_max"] = np.max(mse_values[: len(mse_hist)])
            df.loc[it, "steps"] = mse_values[: len(mse_hist)][
                mse_values > cutoff_steps
            ].size

    return df


def get_coefficient_df(run_ids, expr=False, raw_csv=True):

    df = pd.DataFrame(
        columns=[
            "coeff_run_id",
            "ansatz",
            "qubits",
            "layer_multiplier",
            "seed",
            "coeff_var_abs",
            "coeff_mean_abs",
            "coeff_mean_real",
            "coeff_mean_imag",
            "coeff_var_real",
            "coeff_var_imag",
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
            df.loc[it, "expressibility"] = client.get_run(run_id).data.metrics[
                "expressibility"
            ]
        if raw_csv:
            coeff_df = (
                get_csv_artifact(run_id, "coefficients_filtered")
                .map(lambda s: complex(s))
                .filter(regex="c_.*")
            ).to_numpy()
            df.at[it, "coeff_var_real"] = np.var(np.real(coeff_df), axis=0)
            df.at[it, "coeff_var_imag"] = np.var(np.imag(coeff_df), axis=0)
            df.at[it, "coeff_var_abs"] = np.var(np.abs(coeff_df), axis=0)
            df.at[it, "coeff_mean_real"] = np.mean(np.real(coeff_df), axis=0)
            df.at[it, "coeff_mean_imag"] = np.mean(np.imag(coeff_df), axis=0)
            df.at[it, "coeff_mean_abs"] = np.mean(np.abs(coeff_df), axis=0)

    return df


def get_expressibility_df(run_ids):
    if run_ids is None:
        return None
    df = pd.DataFrame(
        columns=[
            "expr_run_id",
            "ansatz",
            "qubits",
            "seed",
            "expressibility",
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

        df.loc[it, "expressibility"] = client.get_run(run_id).data.metrics[
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
    df = df.sort_values(by=metric, ascending=False)

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
            y=df.expressibility,
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

    fig.update_yaxes(title_text=f"Fourier Coefficient Corr.", secondary_y=False)
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
    df = df.sort_values(by="ansatz", ascending=False)

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
            name=f"{beautify_circuit_name(ansatz)}",
            marker=dict(color=next(main_colors_it), symbol=symbol),
        )

    fig.update_layout(
        title_text="Direct Corr. ({qubit} Qubits, Metric: {metric})",
        template="plotly_white",
        xaxis=dict(
            title=("Corr. Mean" if not weighted else "Weight. Corr. Mean"),
        ),
        yaxis=dict(
            title=metric.title(),
            showgrid=False,
        ),
        showlegend=True,
    )

    return fig


def visualize_expr_scatter(
    df, ansatz_ids, metric, metric_name, weighted=False, legendonly=False
):
    corr_mean = "corr_mean" if not weighted else "corr_w_mean"

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    main_colors_it, sec_colors_it = get_color_iterator(option=0)
    symbols_iterator = get_symbol_iterator(start=5)
    symbols_iterator = iter(
        ["circle", "square", "diamond", "cross", "x", "triangle-up", "hourglass", "star"]
    )
    error_y = False
    error_x = False

    for ansatz_id in ansatz_ids:
        color = next(main_colors_it)
        symbol = next(symbols_iterator)
        _df = df[(df.ansatz_id == ansatz_id)]
        if len(_df) == 0:
            print(f"No data for ansatz_id={ansatz_id}")
            continue
        ansatz = _df["ansatz"].unique()[0]
        if not legendonly:
            fig.add_scattergl(
                x=[_df[metric].mean()],
                y=[_df["expressibility"].mean()],
                error_x=dict(
                    type="data",
                    array=[_df[metric].std()],
                    visible=error_x,
                ),
                error_y=dict(
                    type="data",
                    array=[_df["expressibility"].std()],
                    visible=error_y,
                ),
                mode="markers",
                name=f"{beautify_circuit_name(ansatz)} (EXPR)",
                marker=dict(
                    # color=color,
                    color=design.marker_b_color,
                    # symbol=design.marker_b_style,
                    symbol=symbol,
                    size=design.marker_size,
                    # line=dict(width=design.marker_line_width, color=color),
                    line=dict(
                        width=design.marker_line_width, color=design.marker_b_color
                    ),
                ),
                opacity=design.marker_b_opacity,
                secondary_y=True,
                showlegend=False,
            )

            fig.add_scattergl(
                x=[_df[metric].mean()],
                y=[_df[corr_mean].mean()],
                error_x=dict(
                    type="data",
                    array=[_df[metric].std()],
                    visible=error_x,
                ),
                error_y=dict(
                    type="data",
                    array=[_df[corr_mean].std()],
                    visible=error_y,
                ),
                mode="markers",
                name=f"{beautify_circuit_name(ansatz)} (FCC)",
                marker=dict(
                    # color=color,
                    color=design.marker_a_color,
                    # symbol=design.marker_a_style,
                    symbol=symbol,
                    size=design.marker_size,
                    # line=dict(width=design.marker_line_width, color=color),
                    line=dict(
                        width=design.marker_line_width, color=design.marker_a_color
                    ),
                ),
                opacity=design.marker_a_opacity,
                secondary_y=False,
                showlegend=False,
            )

        fig.add_scattergl(
            x=[None],
            y=[None],
            mode="markers",
            name=beautify_circuit_name(ansatz),
            marker=dict(
                # color=color,
                color=design.legend_color,
                # symbol="triangle-right",
                symbol=symbol,
                size=design.marker_size,
                # line=dict(width=design.marker_line_width, color=color),
                line=dict(width=design.marker_line_width, color=design.legend_color),
            ),
            showlegend=True,
        )
    fig.add_scattergl(
        x=[None],
        y=[None],
        mode="markers",
        name=f"1 - Expressibility",
        marker=dict(
            # color=design.legend_color,
            color=design.marker_b_color,
            # symbol=design.marker_b_style,
            symbol="asterisk",
            size=design.marker_size,
            # line=dict(width=design.marker_line_width, color=design.legend_color),
            line=dict(width=design.marker_line_width, color=design.marker_b_color),
        ),
        showlegend=True,
    )
    fig.add_scattergl(
        x=[None],
        y=[None],
        mode="markers",
        name=f"FCC",
        marker=dict(
            # color=design.legend_color,
            color=design.marker_a_color,
            # symbol=design.marker_a_style,
            symbol="asterisk",
            size=design.marker_size,
            # line=dict(width=design.marker_line_width, color=design.legend_color),
            line=dict(width=design.marker_line_width, color=design.marker_a_color),
        ),
        showlegend=True,
    )

    if not legendonly:
        fig.update_layout(
            title_text="Direct Corr. ({qubit} Qubits, Metric: {metric})",
            template="plotly_white",
            height=500,
            width=680,
            xaxis=dict(
                title=metric_name,
                showgrid=True,
            ),
            yaxis=dict(
                title=(
                    "Fourier Coefficient Corr."
                    if not weighted
                    else "Weight. Fourier Coefficient Corr."
                ),
                anchor="x",
                showgrid=False,
            ),
            yaxis2=dict(
                title=("1 - Expressibility"),
                side="right",
                anchor="x",
                showgrid=False,
            ),
        )

    else:
        fig.update_layout(
            xaxis=dict(
                title="",
                showline=False,
                showgrid=False,
                showticklabels=False,
                zeroline=False,
            ),
            yaxis=dict(
                title="",
                showline=False,
                showgrid=False,
                showticklabels=False,
                zeroline=False,
            ),
            yaxis2=dict(
                title="",
                showline=False,
                showgrid=False,
                showticklabels=False,
                zeroline=False,
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(
                x=0,  # Adjust legend position as needed
                y=1,  # Adjust legend position as needed
                tracegroupgap=20,
                # indention=20,
            ),
        )

    return fig


def calculate_errors(df, ansatz_ids, metric, weighted=False):
    corr_mean = "corr_mean" if not weighted else "corr_w_mean"
    variables = ["expressibility", corr_mean, metric]

    means_by_seed = df.groupby(["seed", "ansatz"], as_index=False)[variables].mean()

    std_across_seeds = means_by_seed.groupby("ansatz")[
        variables
    ].std()  # ddof=1 by default (sample variance)

    result = std_across_seeds.T
    result.index = variables  # give the rows human‑readable names

    print(result)
    return result


def export_pandas_table(result, name, run_ids, experiment_id):
    hs = generate_hash(run_ids)
    path = f"results/{experiment_id}/{hs}/"
    os.makedirs(path, exist_ok=True)
    print(f"Saving csv table to {path}{name}.csv")
    result.to_csv(f"{path}{name}.csv")

    def wrap_num(x):
        # format in scientific notation with 5 significant figures
        return f"\\num{{{x:.1e}}}"

    df_num = result.applymap(wrap_num)

    latex = df_num.to_latex(
        escape=False,  # we already escaped everything we need
        index=True,
        header=True,
        column_format="l"
        + "c" * len(result.columns),  # first column left‑justified, rest centered
        position="htbp",
    )

    print(f"Saving latex table to {path}{name}.tex")
    with open(f"{path}{name}.tex", "w") as f:
        f.write(latex)


def visualize_coeff_variance(df, weighted, legendonly=False):
    pass
    ansaetze = df.ansatz.unique()
    qubit = df["qubits"].unique()[0]

    cols = len(ansaetze)
    # fig = make_subplots(
    #     rows=1,
    #     cols=cols,
    #     subplot_titles=[beautify_circuit_name(ansatz) for ansatz in ansaetze],
    #     horizontal_spacing=0.03,
    #     vertical_spacing=0.03,
    # )
    fig = go.Figure()
    colors = get_color_iterator()[0]
    for col_idx, ansatz in enumerate(ansaetze):
        color = next(colors)
        if not legendonly:
            _df = df[(df.ansatz == ansatz)]

            coeff_var_abs = (
                _df["coeff_var_abs"].apply(lambda x: np.array(eval(x)), 0).mean(axis=0)
            )
            coeff_var_real = (
                _df["coeff_var_real"].apply(lambda x: np.array(eval(x)), 0).mean(axis=0)
            )
            coeff_var_imag = (
                _df["coeff_var_imag"].apply(lambda x: np.array(eval(x)), 0).mean(axis=0)
            )
            coeff_mean_abs = (
                _df["coeff_mean_abs"].apply(lambda x: np.array(eval(x)), 0).mean(axis=0)
            )
            coeff_mean_real = (
                _df["coeff_mean_real"]
                .apply(lambda x: np.array(eval(x)), 0)
                .mean(axis=0)
            )
            coeff_mean_imag = (
                _df["coeff_mean_imag"]
                .apply(lambda x: np.array(eval(x)), 0)
                .mean(axis=0)
            )

            coeff_var_real[coeff_var_abs < 1e-10] = np.nan
            fig.add_trace(
                go.Scatter(
                    y=coeff_var_real,
                    name=f"{beautify_circuit_name(ansatz)}",
                    mode="lines",
                    line=dict(color=color, width=4),
                    showlegend=False,
                    opacity=0.8,
                    marker=dict(
                        # color=color,
                        color=color,
                        # symbol=design.marker_b_style,
                        size=10,
                        # line=dict(width=design.marker_line_width, color=color),
                        line=dict(width=4, color=color),
                    ),
                ),
            )
            coeff_var_imag[coeff_var_imag < 1e-10] = np.nan
            fig.add_trace(
                go.Scatter(
                    y=coeff_var_imag,
                    name=f"{beautify_circuit_name(ansatz)}",
                    mode="markers",
                    line=dict(color=color, width=4),
                    showlegend=False,
                    opacity=0.8,
                    marker=dict(
                        # color=color,
                        color=color,
                        # symbol=design.marker_b_style,
                        size=10,
                        # line=dict(width=design.marker_line_width, color=color),
                        line=dict(width=4, color=color),
                    ),
                ),
            )

        fig.add_scattergl(
            x=[None],
            y=[None],
            mode="markers+lines",
            name=f"{beautify_circuit_name(ansatz)}",
            line=dict(color=color, width=4),
            marker=dict(
                # color=color,
                color=color,
                # symbol=design.marker_b_style,
                size=10,
                # line=dict(width=design.marker_line_width, color=color),
                line=dict(width=4, color=color),
            ),
            showlegend=True,
        )

    if not legendonly:
        fig.update_layout(
            title_text=(
                f"Variance of coefficients for different circuits ({qubit} Qubits)"
            ),
            template="plotly_white",
            height=400,
            width=550,
            # margin_pad=6,
            coloraxis=dict(
                colorscale=design.colorscale,
                colorbar=dict(tickangle=design.hm_tickangle),
            ),
            xaxis=dict(
                title="Frequency",
                showgrid=True,
            ),
            yaxis=dict(
                title=("Coefficient Variance"),
                anchor="x",
                showgrid=False,
                type="log",
            ),
            legend=dict(
                x=1.15,  # Adjust legend position as needed
                y=0.5,  # Adjust legend position as needed
                tracegroupgap=20,
                # indention=20,
            ),
        )
    else:
        fig.update_layout(
            xaxis=dict(
                title="",
                showline=False,
                showgrid=False,
                showticklabels=False,
                zeroline=False,
            ),
            yaxis=dict(
                title="",
                showline=False,
                showgrid=False,
                showticklabels=False,
                zeroline=False,
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(
                x=0,  # Adjust legend position as needed
                y=1,  # Adjust legend position as needed
                tracegroupgap=20,
                # indention=20,
            ),
        )

    return fig


def visualize_heatmap(df, selected_seed, weighted, parameters=False):
    ansaetze = df.ansatz.unique()
    qubit = df["qubits"].unique()[0]

    rows = 2
    cols = len(ansaetze) // rows

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[beautify_circuit_name(ansatz) for ansatz in ansaetze],
        horizontal_spacing=0.01,
        vertical_spacing=0.04,
    )

    for it, ansatz in enumerate(ansaetze):
        _df = df[(df.ansatz == ansatz) & (df.seed == selected_seed)]
        if len(_df) == 0:
            print(
                f"No data for q={qubit}, ansatz={beautify_circuit_name(ansatz)}, seed={selected_seed}"
            )
            continue

        if not parameters:
            sub_fig_trace = get_plotly_heatmap(
                _df.coeff_run_id.item(),
                f"coefficients_correlated{'_weighted' if weighted else ''}",
                automax=True,
            )
        else:
            sub_fig_trace = get_plotly_heatmap(
                _df.coeff_run_id.item(),
                f"parameters_coefficients_correlated",
                automax=True,
            )
        row_idx = 1 if it < cols else rows
        col_idx = (it % cols) + 1

        fig.add_trace(sub_fig_trace, row=row_idx, col=col_idx)

        fig.update_xaxes(
            dict(
                title="Coefficients" if row_idx == rows else "",
                showticklabels=True if row_idx == rows else False,
                tickvals=sub_fig_trace.x,
                ticktext=tickval_to_tex(sub_fig_trace.x),
                tickangle=design.hm_tickangle,
                # automargin=True,
            ),
            showgrid=False,
            row=row_idx,
            col=col_idx,
        )
        fig.update_yaxes(
            dict(
                title="Coefficients" if it % cols == 0 else "",
                showticklabels=True if col_idx == 1 else False,
                autorange="reversed",
                tickvals=sub_fig_trace.y,
                ticktext=tickval_to_tex(sub_fig_trace.y),
                scaleanchor="x",
                tickangle=design.hm_tickangle,
            ),
            showgrid=False,
            row=row_idx,
            col=col_idx,
            # automargin=True,
        )
    fig.update_annotations(yshift=-10)
    fig.update_layout(
        title_text=(
            f"{'Weight. ' if weighted else ''}Corr. of Coefficients for Different Ansaetze ({qubit} Qubits)"
        ),
        template="plotly_white",
        height=300 * rows,
        width=250 * cols,
        margin_pad=6,
        coloraxis=dict(
            colorscale=design.colorscale, colorbar=dict(tickangle=design.hm_tickangle)
        ),
    )

    return fig


def visualize_distribution(df, identifier="fig_distribution_valid"):
    ansaetze = df.ansatz.unique()
    qubit = df["qubits"].unique()[0]
    seeds = df.seed.unique()

    for it, ansatz in enumerate(ansaetze):
        classical_traces = {"Ground Truth": [], "Prediction": [], "Differences": []}
        quantum_traces = {"Ground Truth": [], "Prediction": [], "Differences": []}

        for seed in track(
            seeds, description="Collecting distribution data..", total=len(seeds)
        ):
            _df = df[(df.ansatz == ansatz) & (df.seed == seed)]
            if len(_df) == 0:
                print(
                    f"No data for q={qubit}, ansatz={beautify_circuit_name(ansatz)}, seed={seed}"
                )
                continue

            training_run_ids = ast.literal_eval(_df.training_run_id.unique()[0])
            classical_training_run_ids = ast.literal_eval(
                _df.classical_training_run_id.unique()[0]
            )

            for training_run_id in training_run_ids:
                fig = get_plotly_figure(training_run_id, identifier=identifier)
                if fig is None:
                    print(
                        f"No data for q={qubit}, ansatz={beautify_circuit_name(ansatz)}, seed={seed}, training_id={training_run_id}"
                    )
                    continue
                for dp in fig.data:
                    if dp.type == "histogram":
                        quantum_traces[dp.name].append(dp.x)

            for classical_training_run_id in classical_training_run_ids:
                fig = get_plotly_figure(
                    classical_training_run_id, identifier=identifier
                )
                if fig is None:
                    print(
                        f"No data for q={qubit}, ansatz={beautify_circuit_name(ansatz)}, seed={seed}, training_id={training_run_id}"
                    )
                    continue
                for dp in fig.data:
                    if dp.type == "histogram":
                        classical_traces[dp.name].append(dp.x)
        ground_truth_classical = np.mean(classical_traces["Ground Truth"], axis=0)
        ground_truth_quantum = np.mean(quantum_traces["Ground Truth"], axis=0)
        prediction_quantum = np.mean(quantum_traces["Prediction"], axis=0)
        prediction_classical = np.mean(classical_traces["Prediction"], axis=0)
        differences_quantum = np.mean(quantum_traces["Differences"], axis=0)
        differences_classical = np.mean(classical_traces["Differences"], axis=0)

        fig = ff.create_distplot(
            [differences_quantum, differences_classical],
            [
                f"QFM:<br>μ={differences_quantum.mean():.3f}<br>σ={differences_quantum.std():.2f}",
                f"MLP:<br>μ={differences_classical.mean():.3f}<br>σ={differences_classical.std():.2f}",
            ],
            colors=[design.marker_a_color, design.marker_b_color],
            bin_size=2,
            curve_type="kde",
            show_rug=False,
        )

        fig.update_layout(
            title_text=f"Ground Truth and Model Prediction and Differences",
            plot_bgcolor="rgba(0,0,0,0)",
            template="plotly_white",
            xaxis=dict(
                title="Absolute difference of transverse momenta",
                showgrid=False,
            ),
            yaxis=dict(
                title="Frequency",
                showgrid=False,
            ),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
    return fig


def visualize_coeff_param_relation(df, selected_seed):
    ansaetze = df.ansatz.unique()
    qubit = df["qubits"].unique()[0]

    rows = 4
    cols = len(ansaetze) // (rows // 2)

    # first half of ansatz names, then one line empty, then second half and one line empty
    titles = []
    for r in range(rows):
        for c in range(cols):
            if r == 0:
                titles.append(ansaetze[c].replace("_", " "))
            elif r == 2:
                titles.append(ansaetze[len(ansaetze) // 2 + c].replace("_", " "))
            else:
                titles.append("")
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.01,
    )

    for it, ansatz in enumerate(ansaetze):
        _df = df[(df.ansatz == ansatz) & (df.seed == selected_seed)]
        if len(_df) == 0:
            print(
                f"No data for q={qubit}, ansatz={beautify_circuit_name(ansatz)}, seed={selected_seed}"
            )
            continue

        sub_fig_trace = get_plotly_figure(
            _df.coeff_run_id.item(),
            f"parameters_coefficients_complex",
        )
        col_idx = (it % cols) + 1
        row_idx = 1 if it < cols else 3

        fig.add_traces(
            sub_fig_trace.data,
            cols=col_idx,
            rows=[row_idx, row_idx + 1] * (len(sub_fig_trace.data) // 2),
        )

        if row_idx == 3:
            fig.update_xaxes(
                dict(
                    title="",
                    showticklabels=False,
                    showgrid=True,
                ),
                showgrid=False,
                row=row_idx,
                col=col_idx,
            )
            fig.update_xaxes(
                dict(
                    title="Parameters",
                    showticklabels=True,
                    showgrid=True,
                    range=[0, 2 * np.pi],
                    tickmode="array",
                    tickvals=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                    ticktext=[
                        r"$0$",
                        r"$\frac{\pi}{2}$",
                        r"$\pi$",
                        r"$\frac{3\pi}{2}$",
                        r"$2\pi$",
                    ],
                ),
                showgrid=False,
                row=row_idx + 1,
                col=col_idx,
            )
        else:
            fig.update_xaxes(
                dict(
                    title="",
                    showticklabels=False,
                    showgrid=True,
                ),
                showgrid=False,
                row=row_idx,
                col=col_idx,
            )
            fig.update_xaxes(
                dict(
                    title="",
                    showticklabels=False,
                    showgrid=True,
                ),
                showgrid=False,
                row=row_idx + 1,
                col=col_idx,
            )
        fig.update_yaxes(
            dict(
                title="Absolute" if it % cols == 0 else "",
                showticklabels=False if col_idx == 1 else False,
                showgrid=False,
                type="log",
            ),
            showgrid=False,
            row=row_idx,
            col=col_idx,
            # automargin=True,
        )
        fig.update_yaxes(
            dict(
                title="Phase" if it % cols == 0 else "",
                showticklabels=False if col_idx == 1 else False,
                showgrid=False,
            ),
            showgrid=False,
            row=row_idx + 1,
            col=col_idx,
            # automargin=True,
        )
        fig.update_traces(showlegend=False, row=row_idx, col=col_idx)
        fig.update_traces(showlegend=False, row=row_idx + 1, col=col_idx)

    main_colors_it = iter(plotly.colors.qualitative.Dark2)
    for j in range(qubit):
        fig.add_scattergl(
            x=[None],
            y=[None],
            mode="markers",
            name=f"c_{j}",
            marker=dict(
                # color=design.legend_color,
                color=next(main_colors_it),
            ),
            showlegend=True,
        )
    # fig.update_annotations(yshift=-10)
    fig.update_layout(
        title_text=(f"Parameter - Coefficient Relation for {qubit} Qubits"),
        template="plotly_white",
        height=220 * rows,
        width=280 * cols,
    )
    return fig


def visualize_single_heatmap(df, selected_seed, identifier="coefficients_correlated"):
    qubit = df["qubits"].unique()[0]

    fig = go.Figure()

    _df = df[(df.seed == selected_seed)]
    if len(_df) == 0:
        print(f"No data for q={qubit}, seed={selected_seed}")
        return fig

    # sub_fig_trace = get_plotly_heatmap(
    #     _df.coeff_run_id.item(),
    #     identifier,
    #     automax=True,
    # )

    fig = get_plotly_figure(_df.coeff_run_id.item(), identifier)

    # fig.add_trace(sub_fig_trace)
    fig.update_xaxes(
        dict(
            title="Coefficients",
            tickvals=fig.data[0].x,
            ticktext=tickval_to_tex(fig.data[0].x),
            tickangle=design.hm_tickangle,
            # automargin=True,
        ),
        showgrid=False,
    )
    fig.update_yaxes(
        dict(
            title="Coefficients",
            autorange="reversed",
            tickvals=fig.data[0].y,
            ticktext=tickval_to_tex(fig.data[0].y),
            scaleanchor="x",
            tickangle=design.hm_tickangle,
            # automargin=True,
        ),
        showgrid=False,
    )

    fig.update_annotations(yshift=-10)
    fig.update_layout(
        title_text=(f"Corr. of Coefficients for Different Ansaetze ({qubit} Qubits)"),
        template="plotly_white",
        height=100 * qubit,
        width=110 * qubit,
        margin_pad=4,
        coloraxis=dict(
            colorscale=design.colorscale, colorbar=dict(tickangle=design.hm_tickangle)
        ),
    )

    return fig
