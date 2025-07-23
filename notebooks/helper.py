import plotly
from plotly.validators.scatter.marker import SymbolValidator
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

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# pio.kaleido.scope.mathjax = None


class design:
    marker_size = 14
    marker_line_width = 1
    marker_a_opacity = 1.0
    marker_b_opacity = 1.0
    marker_a_style = "cross"
    marker_a_color = "#009682"
    marker_b_style = "x"
    marker_b_color = "#DF9B1B"
    legend_color = "#002D4C"
    colorscale = "Sunset"
    annotation_font_offset = 1
    tick_font_offset = 1
    large_tick_font_offset = 0
    hm_tickangle = 0


def save_fig(
    fig,
    name,
    run_ids,
    experiment_id,
    font_size=16,
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
    raw_symbols = SymbolValidator().values
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


def get_plotly_distribution(run_id, identifier="fig_distribution_train"):
    client = mlflow.tracking.MlflowClient()
    try:
        fig_path = client.download_artifacts(run_id, f"{identifier}.html", "./")
    except FileNotFoundError:
        print(f"File {fig_path} not found for run id {run_id}")
        return None
    fig = read_from_html(fig_path)

    os.remove(fig_path)

    return fig


def rgb_to_rgba(rgb_value: str, alpha: float):
    """
    Adds the alpha channel to an RGB Value and returns it as an RGBA Value
    :param rgb_value: Input RGB Value
    :param alpha: Alpha Value to add  in range [0,1]
    :return: RGBA Value
    """
    return f"rgba{rgb_value[3:-1]}, {alpha})"


def get_training_df(run_ids, cutoff_mse=-1, cutoff_steps=-1, metric="mse"):
    if run_ids is None:
        return None
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


def get_classical_training_df(run_ids, cutoff_mse=-1, cutoff_steps=-1, metric="mse"):
    if run_ids is None:
        return None
    df = pd.DataFrame(
        columns=[
            "training_run_id",
            "width",
            "depth",
            "seed",
            f"{metric}",
            f"{metric}_min",
            f"{metric}_max",
            "steps",
        ]
    )

    for it, run_id in track(
        enumerate(run_ids),
        description="Collecting classical training data..",
        total=len(run_ids),
    ):
        client = mlflow.tracking.MlflowClient()
        if client.get_run(run_id).info.status != "FINISHED":
            print(f"Run {run_id} not finished")
            continue

        df.loc[it, "training_run_id"] = run_id
        df.loc[it, "width"] = int(client.get_run(run_id).data.params["model.width"])
        df.loc[it, "depth"] = int(client.get_run(run_id).data.params["model.depth"])
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


def visualize_expr_scatter(df, ansatz_ids, metric, weighted=False, legendonly=False):
    corr_mean = "corr_mean" if not weighted else "corr_w_mean"

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    main_colors_it, sec_colors_it = get_color_iterator(option=0)
    symbols_iterator = get_symbol_iterator(start=5)
    symbols_iterator = iter(
        ["circle", "square", "diamond", "cross", "x", "triangle-up", "hexagon", "star"]
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
                y=[_df["kl_divergence"].mean()],
                error_x=dict(
                    type="data",
                    array=[_df[metric].std()],
                    visible=error_x,
                ),
                error_y=dict(
                    type="data",
                    array=[_df["kl_divergence"].std()],
                    visible=error_y,
                ),
                mode="markers",
                name=f"{ansatz} (EXPR)",
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
                name=f"{ansatz} (FCC)",
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
            name=f"{ansatz}",
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
        name=f"Expressibility",
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
            title_text="Direct Correlation ({qubit} Qubits, Metric: {metric})",
            template="plotly_white",
            xaxis=dict(
                title="Mean Squared Error",
                showgrid=True,
            ),
            yaxis=dict(
                title=(
                    "Fourier Coefficient Correlation"
                    if not weighted
                    else "Weighted Fourier Coefficient Correlation"
                ),
                anchor="x",
                showgrid=False,
            ),
            yaxis2=dict(
                title=("Expressibility"),
                side="right",
                anchor="x",
                showgrid=False,
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


def visualize_heatmap(df, selected_seed, weighted, parameters=False):
    ansaetze = df.ansatz.unique()
    qubit = df["qubits"].unique()[0]

    rows = 2
    cols = len(ansaetze) // rows

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[ansatz.replace("_", " ") for ansatz in ansaetze],
        horizontal_spacing=0.01,
        vertical_spacing=0.03,
    )

    for it, ansatz in enumerate(ansaetze):
        _df = df[(df.ansatz == ansatz) & (df.seed == selected_seed)]
        if len(_df) == 0:
            print(f"No data for q={qubit}, ansatz={ansatz}, seed={selected_seed}")
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
            f"{'Weighted ' if weighted else ''}Correlation of Coefficients for Different Ansaetze ({qubit} Qubits)"
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


def visualize_distribution(df):
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
                print(f"No data for q={qubit}, ansatz={ansatz}, seed={seed}")
                continue

            training_run_ids = ast.literal_eval(_df.training_run_id.unique()[0])
            classical_training_run_ids = ast.literal_eval(
                _df.classical_training_run_id.unique()[0]
            )

            for training_run_id in training_run_ids:
                fig = get_plotly_distribution(training_run_id)
                if fig is None:
                    print(
                        f"No data for q={qubit}, ansatz={ansatz}, seed={seed}, training_id={training_run_id}"
                    )
                    continue
                for dp in fig.data:
                    if dp.type == "histogram":
                        quantum_traces[dp.name].append(dp.x)

            for classical_training_run_id in classical_training_run_ids:
                fig = get_plotly_distribution(classical_training_run_id)
                if fig is None:
                    print(
                        f"No data for q={qubit}, ansatz={ansatz}, seed={seed}, training_id={training_run_id}"
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
                f"QFM: μ={differences_quantum.mean():.3f}, σ={differences_quantum.std():.2f}",
                f"MLP: μ={differences_classical.mean():.3f}, σ={differences_classical.std():.2f}",
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


def visualize_single_heatmap(df, selected_seed, selected_ansatz, weighted):
    qubit = df["qubits"].unique()[0]

    fig = go.Figure()

    _df = df[(df.ansatz == selected_ansatz) & (df.seed == selected_seed)]
    if len(_df) == 0:
        print(f"No data for q={qubit}, ansatz={selected_ansatz}, seed={selected_seed}")
        return fig

    sub_fig_trace = get_plotly_heatmap(
        _df.coeff_run_id.item(),
        f"coefficients_correlated{'_weighted' if weighted else ''}",
        automax=True,
    )

    fig.add_trace(sub_fig_trace)
    fig.update_xaxes(
        dict(
            title="Coefficients",
            tickvals=sub_fig_trace.x,
            ticktext=tickval_to_tex(sub_fig_trace.x),
            tickangle=design.hm_tickangle,
            # automargin=True,
        ),
        showgrid=False,
    )
    fig.update_yaxes(
        dict(
            title="Coefficients",
            autorange="reversed",
            tickvals=sub_fig_trace.y,
            ticktext=tickval_to_tex(sub_fig_trace.y),
            scaleanchor="x",
            tickangle=design.hm_tickangle,
            # automargin=True,
        ),
        showgrid=False,
    )

    fig.update_layout(
        title_text=(
            f"{'Weighted ' if weighted else ''}Correlation of Coefficients for Different Ansaetze ({qubit} Qubits)"
        ),
        template="plotly_white",
        height=100 * qubit,
        width=110 * qubit,
        margin_pad=4,
        coloraxis=dict(
            colorscale=design.colorscale, colorbar=dict(tickangle=design.hm_tickangle)
        ),
    )

    return fig
