from qml_essentials.model import Model
from torch.utils.data import DataLoader
import torch
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
import pandas as pd
import mlflow
import plotly.express as px
import logging
import scipy
from qml_essentials import coefficients as QMLCoefficients


from typing import Dict, List

log = logging.getLogger(__name__)


def rgb_to_rgba(rgb_value: str, alpha: float):
    """
    Adds the alpha channel to an RGB Value and returns it as an RGBA Value
    :param rgb_value: Input RGB Value
    :param alpha: Alpha Value to add  in range [0,1]
    :return: RGBA Value
    """
    return f"rgba{rgb_value[3:-1]}, {alpha})"


def visualize_heatmap_filtered(
    df: pd.DataFrame,
    zmax=1.0,
    zmin=0.0,
) -> go.Figure:

    fig = go.Figure(
        data=go.Heatmap(
            z=df,
            y=df.index,
            x=df.columns,
            hoverongaps=False,
            colorscale="Sunset",
            zmax=zmax,
            zmin=zmin,
        )
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        width=800,
        height=800,
        autosize=False,
    )
    return fig


def visualize_coefficients_decay(df: pd.DataFrame, model: Model) -> go.Figure:
    if model.n_input_feat == 1:
        fig = go.Figure(data=go.Scatter(x=df.index, y=df, mode="markers+lines"))
        fig.update_layout(
            template="plotly_white",
            title="Coefficient Mean",
            yaxis_type="log",
        )
    elif model.n_input_feat == 2:
        heatmap = pd.DataFrame(
            df.to_numpy().reshape(model.degree + 1, model.degree + 1),
            columns=["c_+" + str(i) for i in range(model.degree + 1)],
            index=["c_+" + str(i) for i in range(model.degree + 1)],
        )
        fig = visualize_heatmap_filtered(df=heatmap, zmax=heatmap.max(axis=None))
    else:
        raise NotImplementedError(
            "Only implemented for n_input_feat=1 and n_input_feat=2"
        )

    return fig


def visualize_data_spectrum(df: pd.DataFrame, model: Model, mts, mfs) -> go.Figure:
    df_filtered = df.filter(regex="c.*", axis=1)

    if model.n_input_feat == 1:

        fig = go.Figure()
        for c in df_filtered.columns:
            fig.add_trace(
                go.Box(
                    y=df_filtered[c],
                    name=c,
                    marker=dict(color=pc.qualitative.Dark2[0]),
                    boxpoints=False,
                )
            )
        fig.update_layout(
            template="plotly_white",
            showlegend=False,
            xaxis=dict(
                title="X1",
                showgrid=False,
            ),
            yaxis=dict(
                showgrid=False,
            ),
        )
    elif model.n_input_feat == 2:
        n_coeffs = int(np.sqrt(df.size))
        n_freqs = n_coeffs // 2

        coeffs = df.mean(axis=0).to_numpy().reshape((n_coeffs, n_coeffs))

        fig = go.Figure(
            data=go.Heatmap(
                z=np.log(np.abs(coeffs)),
                hoverongaps=False,
                colorscale="Sunset",
            )
        )
        n_freqs: int = 2 * mfs * model.degree + 1
        freqs = np.fft.fftshift(np.fft.fftfreq(mts * n_freqs, 1 / n_freqs))

        def str_sign(num: int):
            return f"{num:.2f}" if num < 0 else f"+{num:.2f}"

        fig.update_xaxes(
            tickvals=list(range(n_coeffs)),
            ticktext=[str_sign(f) for f in freqs],
            title="X2",
        )
        fig.update_yaxes(
            tickvals=list(range(n_coeffs)),
            ticktext=[str_sign(f) for f in freqs],
            title="X1",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            width=800,
            height=800,
            autosize=False,
        )
    else:
        raise NotImplementedError(
            "Only implemented for n_input_feat=1 and n_input_feat=2"
        )

    return fig


def visualize_model_spectrum(df: pd.DataFrame, model: Model, mts, mfs) -> go.Figure:
    coeffs, freqs = QMLCoefficients.Coefficients.get_spectrum(
        model, shift=True, trim=True
    )

    if model.n_input_feat == 1:

        fig = go.Figure(
            data=[
                go.Bar(
                    x=freqs.flatten(),
                    y=np.abs(coeffs).flatten(),
                    marker=dict(color=pc.qualitative.Dark2[0]),
                    name="Prediction",
                ),
                go.Bar(
                    x=freqs.flatten(),
                    y=-np.abs(df.to_numpy()).flatten(),
                    marker=dict(color=pc.qualitative.Dark2[1]),
                    name="Ground Truth",
                ),
            ]
        )
        fig.update_layout(
            template="plotly_white",
            title="Spectrum",
            showlegend=True,
            xaxis=dict(
                title="X1",
                showgrid=False,
            ),
            yaxis=dict(
                showgrid=False,
            ),
        )
    elif model.n_input_feat == 2:
        fig = go.Figure(
            data=go.Heatmap(
                colorscale="Sunset",
                name="Prediction",
            )
        )
        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(
            go.Heatmap(
                z=np.abs(df.to_numpy().reshape(coeffs.shape)),
                colorscale="Sunset",
                name="Ground Truth",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                z=np.abs(coeffs),
                colorscale="Sunset",
                name="Prediction",
            ),
            row=1,
            col=2,
        )

        def str_sign(num: int):
            return f"{num:.2f}" if num < 0 else f"+{num:.2f}"

        for col in [1, 2]:
            fig.update_xaxes(
                tickvals=list(range(coeffs.shape[1])),
                ticktext=[str_sign(f) for f in freqs],
                title="X2",
                row=1,
                col=col,
            )
            fig.update_yaxes(
                tickvals=list(range(coeffs.shape[0])),
                ticktext=[str_sign(f) for f in freqs],
                title="X1",
                row=1,
                col=col,
            )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            width=1600,
            height=800,
            autosize=False,
        )
    else:
        raise NotImplementedError(
            "Only implemented for n_input_feat=1 and n_input_feat=2"
        )

    return {"model": fig}


def visualize_coefficients_correlated(
    df: pd.DataFrame, model: Model, triu=False
) -> go.Figure:

    if triu:
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                if i <= j:
                    df.iloc[i, j] = np.nan
                    # df_weighted.iloc[i, j] = np.nan
        df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
        # df_weighted = df_weighted.dropna(how="all", axis=0).dropna(
        #     how="all", axis=1
        # )

    # TODO: move before triu?
    mlflow.log_metric("coefficients_correlation_variance", df.abs().var().var())
    mlflow.log_metric("coefficients_correlation_mean", df.abs().mean(axis=None))
    mlflow.log_metric("coefficients_correlation_max", df.abs().max(axis=None))
    mlflow.log_metric("coefficients_correlation_min", df.abs().min(axis=None))

    if model.n_input_feat == 1:
        fig = visualize_heatmap_filtered(
            df=df.abs(),
        )

        fig.update_layout(
            title_text=f"Correlated Coefficients for {model.pqc.__class__.__name__}",
            xaxis=dict(
                title="Coefficients",
            ),
            yaxis=dict(title="Coefficients", autorange="reversed", scaleanchor="x"),
        )
    elif model.n_input_feat == 2:
        fig = visualize_heatmap_filtered(
            df=df.abs(),
        )

        fig.update_layout(
            title_text=f"Correlated Coefficients for {model.pqc.__class__.__name__}",
            xaxis=dict(
                title="Coefficients",
            ),
            yaxis=dict(title="Coefficients", autorange="reversed", scaleanchor="x"),
        )
    else:
        raise NotImplementedError(
            "Only implemented for n_input_feat=1 and n_input_feat=2"
        )

    return fig


def visualize_coefficients_correlated_complex(
    df: pd.DataFrame, model: Model, triu=False
) -> go.Figure:

    if triu:
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                if i <= j:
                    df.iloc[i, j] = np.nan
                    # df_weighted.iloc[i, j] = np.nan
        df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
        # df_weighted = df_weighted.dropna(how="all", axis=0).dropna(
        #     how="all", axis=1
        # )

    if model.n_input_feat == 1:
        fig = visualize_heatmap_filtered(
            df=df.abs(),
        )

        fig.update_layout(
            title_text=f"Correlated Coefficients for {model.pqc.__class__.__name__}",
            xaxis=dict(
                title="Coefficients",
            ),
            yaxis=dict(title="Coefficients", autorange="reversed", scaleanchor="x"),
        )
    elif model.n_input_feat == 2:
        fig = visualize_heatmap_filtered(
            df=df.abs(),
        )

        fig.update_layout(
            title_text=f"Correlated Coefficients for {model.pqc.__class__.__name__}",
            xaxis=dict(
                title="Coefficients",
            ),
            yaxis=dict(title="Coefficients", autorange="reversed", scaleanchor="x"),
        )
    else:
        raise NotImplementedError(
            "Only implemented for n_input_feat=1 and n_input_feat=2"
        )

    return fig


def visualize_coefficients_correlated_3d(
    df: pd.DataFrame, model: Model, triu=False
) -> go.Figure:

    # if triu:
    #     for i in range(df.shape[0]):
    #         for j in range(df.shape[1]):
    #             if i <= j:
    #                 df.iloc[i, j] = np.nan
    #                 # df_weighted.iloc[i, j] = np.nan
    #     df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    # df_weighted = df_weighted.dropna(how="all", axis=0).dropna(
    #     how="all", axis=1
    # )

    mlflow.log_metric("coefficients_correlation_variance", df.abs().var().var())
    mlflow.log_metric("coefficients_correlation_mean", df.abs().mean(axis=None))
    mlflow.log_metric("coefficients_correlation_max", df.abs().max(axis=None))
    mlflow.log_metric("coefficients_correlation_min", df.abs().min(axis=None))

    if model.n_input_feat == 1:
        fig = visualize_heatmap_filtered(
            df=df,
        )

        fig.update_layout(
            title_text=f"Correlated Coefficients for {model.pqc.__class__.__name__}",
            xaxis=dict(
                title="Coefficients",
            ),
            yaxis=dict(title="Coefficients", autorange="reversed", scaleanchor="x"),
        )
    elif model.n_input_feat == 2:
        Y = []
        layer_idx = 0
        for X1 in range(model.degree + 1):
            Y_temp = (
                df.filter(regex=rf"c(_\+{X1}_\+)", axis=0)
                .filter(regex=rf"c(_\+{X1}_\+)", axis=1)
                .to_numpy()
            )
            Y_temp -= np.eye(model.degree + 1) - layer_idx
            layer_idx += 1
            Y.append(Y_temp)
        for X2 in range(model.degree + 1):
            Y_temp = (
                df.filter(regex=rf"c(_\+{X1}_\+)", axis=0)
                .filter(regex=rf"c(_\+{X1}_\+)", axis=1)
                .to_numpy()
            )
            Y_temp -= np.eye(model.degree + 1) - layer_idx
            layer_idx += 1
            Y.append(Y_temp)
        fig = go.Figure(
            data=[
                go.Surface(z=Y[i])
                for i in range(model.n_input_feat * (model.degree + 1))
            ]
        )

        fig.update_layout(
            title_text=f"{model.pqc.__class__.__name__}",
            xaxis=dict(title="X1"),
            yaxis=dict(title="X2"),
            # zaxis=dict(title="Correlation"),
            plot_bgcolor="rgba(0,0,0,0)",
            width=800,
            height=800,
            autosize=False,
        )
    else:
        raise NotImplementedError(
            "Only implemented for n_input_feat=1 and n_input_feat=2"
        )

    return fig


def visualize_coefficients_correlated_filtered_weighted(
    df: pd.DataFrame, model: Model, triu=False
) -> go.Figure:

    if triu:
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                if i <= j:
                    df.iloc[i, j] = np.nan
                    # df_weighted.iloc[i, j] = np.nan
        df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
        # df_weighted = df_weighted.dropna(how="all", axis=0).dropna(
        #     how="all", axis=1
        # )

    mlflow.log_metric(
        "coefficients_correlation_weighted_variance", df.abs().var().var()
    )
    mlflow.log_metric(
        "coefficients_correlation_weighted_mean", df.abs().mean(axis=None)
    )
    mlflow.log_metric("coefficients_correlation_weighted_max", df.abs().max(axis=None))
    mlflow.log_metric("coefficients_correlation_weighted_min", df.abs().min(axis=None))

    fig = visualize_heatmap_filtered(
        df=df,
    )

    fig.update_layout(
        title_text=f"Weighted Correlated Coefficients for {model.pqc.__class__.__name__}",
        xaxis=dict(
            title="Coefficients",
        ),
        yaxis=dict(title="Coefficients", autorange="reversed", scaleanchor="x"),
    )

    return fig


def visualize_parameters_correlated(
    df: pd.DataFrame, model: Model, triu=False
) -> go.Figure:
    df = df.filter(regex="p.*", axis=0).filter(regex="p.*", axis=1)

    if triu:
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                if i <= j:
                    df.iloc[i, j] = np.nan
        df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)

    mlflow.log_metric("parameters_correlation_variance", df.var().var())
    mlflow.log_metric("parameters_correlation_mean", df.mean(axis=None))
    mlflow.log_metric("parameters_correlation_max", df.max(axis=None))
    mlflow.log_metric("parameters_correlation_min", df.min(axis=None))

    fig = visualize_heatmap_filtered(
        df=df,
    )
    fig.update_layout(
        title_text=f"Correlated Parameters for {model.pqc.__class__.__name__}",
        xaxis=dict(
            title="Parameters",
        ),
        yaxis=dict(title="Parameters", autorange="reversed", scaleanchor="x"),
    )

    return fig


def visualize_parameters_coefficients_correlated(
    df: pd.DataFrame, model: Model, discard_negative=True
) -> go.Figure:
    df_filtered = df.filter(regex=r"p_\d+", axis=0).filter(
        regex=rf"c(_\+\d+){{{model.n_input_feat}}}", axis=1
    )

    mlflow.log_metric(
        "parameters_coefficients_correlation_variance", df_filtered.var().var()
    )
    mlflow.log_metric(
        "parameters_coefficients_correlation_mean", df_filtered.mean(axis=None)
    )
    mlflow.log_metric(
        "parameters_coefficients_correlation_max", df_filtered.max(axis=None)
    )
    mlflow.log_metric(
        "parameters_coefficients_correlation_min", df_filtered.min(axis=None)
    )

    fig = visualize_heatmap_filtered(
        df=df_filtered.abs(), zmax=df_filtered.max(axis=None)
    )
    fig.update_layout(
        title_text=f"Correlation Parameters Coefficients for {model.pqc.__class__.__name__}",
        xaxis=dict(
            title="Coefficients",
        ),
        yaxis=dict(title="Parameters", autorange="reversed", scaleanchor="x"),
    )

    return fig


def visualize_parameters_coefficients_complex(
    df: pd.DataFrame, model: Model, discard_negative=True
) -> go.Figure:
    df_filtered = df.filter(regex=rf"(_\+?\d+){{{model.n_input_feat}}}", axis=1)

    fig = make_subplots(
        rows=model.params.size,
        cols=model.degree,
        shared_xaxes=True,
        shared_yaxes=True,
        column_titles=[f"c_{j}" for j in range(model.degree)],
        row_titles=[f"p_{j}" for j in range(model.params.size)],
        vertical_spacing=0.01,
        horizontal_spacing=0.01,
    )

    def phase_to_color(phase):
        n_samples = 360  # 1 color per degree
        # sample colormap
        colormap = np.array(pc.sample_colorscale(pc.cyclical.Phase, n_samples + 1))
        # discretize array to n_samples
        phase_bins = np.linspace(0, np.pi, n_samples)
        # map phase to colormap
        inds = np.digitize(phase % np.pi, phase_bins)
        return colormap[inds]

    max_c = [0 for _ in range(model.degree)]
    for i in range(model.params.size):
        for j in range(model.degree):
            absolute = np.abs(df_filtered[f"c_+{j}"])
            phase = np.angle(df_filtered[f"c_+{j}"])

            if absolute.max() > max_c[j]:
                max_c[j] = absolute.max()
            fig.add_trace(
                go.Scatter(
                    x=absolute.to_list(),
                    y=df_filtered[f"p_{i}"].to_list(),
                    mode="markers",
                    name=f"c_{j}p_{j}",
                    marker=dict(size=2, color=phase_to_color(phase)),
                    showlegend=False,
                    xaxis="x",
                    yaxis="y",
                ),
                row=i + 1,
                col=j + 1,
            )

    fig.update_layout(
        # title_text=f"Correlation Parameters Coefficients for {model.pqc.__class__.__name__}",
        height=100 * model.params.size,
        width=110 * model.degree,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_white",
        margin={"l": 25, "r": 25, "t": 25, "b": 25},
    )
    for i in range(1, model.params.size + 1):
        fig.update_yaxes(
            range=[0, 2 * np.pi],
            tickvals=[f"{0}", f"{np.pi:.2f}", f"{2*np.pi:.2f}"],
            row=i,
        )
    for i in range(1, model.degree + 1):
        fig.update_xaxes(
            range=[0, max_c[i - 1]],
            tickvals=[f"{0}", f"{0.9*max_c[i - 1]:.2f}"],
            col=i,
        )

    return fig


def visualize_parameters_coefficients_dist(
    df: pd.DataFrame, model: Model, discard_negative=True
) -> go.Figure:
    df_filtered = df.filter(regex=rf"(_\+?\d+){{{model.n_input_feat}}}", axis=1)

    def two_d_gaussian(x, *args):
        # args should contain multiple of a, mu, sigma
        # where a is the amplitude, mu is the mean, and sigma is the standard deviation
        # such that f(x)=a {\frac {1}{\sigma {\sqrt {2\pi }}}}\exp \left(-{\frac {1}{2}}{\frac {(x-\mu )^{2}}{\sigma ^{2}}}\right)
        # for 1 gaussian
        result = 0
        for i in range(0, len(args), 3):
            a, mu, sigma = args[i : i + 3]
            result += (
                a
                * (1 / np.sqrt(2 * np.pi * sigma**2))
                * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
            )
        return result

    y_binned_abs = []
    y_binned_angle = []
    for i in range(model.params.size):
        # get all parameter values
        x = df_filtered[f"p_{i}"].to_numpy()
        # create indices for coefficients in bins
        n_bins = 100
        bins = np.linspace(0, 2 * np.pi, n_bins)
        inds = np.digitize(x, bins, right=True)

        y_binned_abs_c = []
        y_binned_angle_c = []
        for j in range(model.degree):
            # get coefficient values
            y = df_filtered[f"c_+{j}"].to_numpy()
            y_abs = np.abs(y)
            y_binned_abs_c.append(
                [np.quantile(y_abs[inds == j], 0.75) for j in range(1, n_bins)]
            )
            y_angle = np.angle(y)
            y_binned_angle_c.append(
                [np.quantile(y_angle[inds == j], 0.25) for j in range(1, n_bins)]
            )

        y_binned_abs.append(y_binned_abs_c)
        y_binned_angle.append(y_binned_angle_c)

    def butter_lowpass_filter(data, cutoff=15, order=2):
        normal_cutoff = cutoff / (2 * n_bins)
        # Get the filter coefficients
        b, a = scipy.signal.butter(order, normal_cutoff, btype="low", analog=False)
        y = scipy.signal.filtfilt(b, a, data)
        return y

    threshold = 1e-6

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        shared_yaxes=False,
        vertical_spacing=0.05,
        # subplot_titles=["Absolute Value", "Phase"],
    )

    for i in range(len(y_binned_abs)):
        main_colors_it = iter(pc.qualitative.Dark2)
        for j in range(model.degree):
            if np.isclose(
                np.mean(y_binned_angle[i][j]), np.pi, rtol=1e-2
            ) or np.isclose(np.mean(y_binned_angle[i][j]), 0, rtol=1e-2):
                y_angle = [None]
            else:
                y_angle = butter_lowpass_filter(y_binned_angle[i][j])
            if np.mean(y_binned_abs[i][j]) < threshold:
                y_abs = [None]
                y_angle = [None]
            else:
                y_abs = butter_lowpass_filter(y_binned_abs[i][j])

            color = rgb_to_rgba(next(main_colors_it), 0.3)
            fig.add_trace(
                go.Scatter(
                    x=bins,
                    y=y_abs,
                    mode="lines",
                    name=f"c_{j}_abs",
                    marker=dict(color=color),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=bins,
                    y=y_angle,
                    mode="lines",
                    name=f"c_{j}_angle",
                    marker=dict(color=color),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

    main_colors_it = iter(pc.qualitative.Dark2)
    for j in range(model.degree):
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

    # add log scale
    fig.update_layout(
        title_text=f"Coefficient Parameter Relation",
        plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_white",
        xaxis=dict(
            showgrid=True,
        ),
        xaxis2=dict(
            title="Parameter Value",
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
        yaxis=dict(title="Absolute", showgrid=False, type="log"),
        yaxis2=dict(title="Phase", showgrid=False),
    )
    fig.write_image(f"coeff_param_{model.pqc.__class__.__name__.lower()}.pdf")
    return fig


def visualize_coefficients_correlated_control(
    df: pd.DataFrame,
    model: Model,
) -> go.Figure:
    fig = go.Figure(
        data=go.Scatter(
            x=df["control_value"],
            y=df["coeff_mean"],
        )
    )
    fig.update_layout(
        title_text=f"Correlated Coefficients for {model.pqc.__class__.__name__} over control values",
        plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_white",
        xaxis=dict(
            title="Control Value",
            showgrid=False,
        ),
        yaxis=dict(
            title="Correlation Mean",
            showgrid=False,
        ),
    )
    return fig


def visualize_model(
    model: Model,
    data_loader: DataLoader,
    noise_params: Dict,
) -> go.Figure:
    # access dataset directly, we don't need batches now
    domain_samples = data_loader.dataset.tensors[0]
    fourier_series = data_loader.dataset.tensors[1]

    if model.n_input_feat == 1:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=domain_samples.flatten(),
                    y=fourier_series,
                    mode="lines",
                    name="Ground Truth",
                ),
                go.Scatter(
                    x=domain_samples.flatten(),
                    y=model(
                        params=model.params,
                        inputs=domain_samples,
                        noise_params=noise_params,
                        force_mean=True,
                    ),
                    mode="lines",
                    name="Prediction",
                ),
            ]
        )
        fig.update_layout(
            title_text=f"Ground Truth and Model Prediction",
            plot_bgcolor="rgba(0,0,0,0)",
            template="plotly_white",
            xaxis=dict(
                title="x1",
                showgrid=False,
            ),
            yaxis=dict(
                showgrid=False,
            ),
        )
    elif model.n_input_feat == 2:
        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(
            go.Heatmap(
                x=domain_samples[:, 0],
                y=domain_samples[:, 1],
                z=fourier_series,
                colorscale="Sunset",
                name="Ground Truth",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                x=domain_samples[:, 0],
                y=domain_samples[:, 1],
                z=model(
                    params=model.params,
                    inputs=domain_samples,
                    noise_params=noise_params,
                    force_mean=True,
                ),
                colorscale="Sunset",
                name="Prediction",
            ),
            row=1,
            col=2,
        )
        for col in [1, 2]:
            fig.update_xaxes(
                tickvals=list(range(domain_samples.shape[1])),
                ticktext=domain_samples[:, 1],
                title="x2",
                row=1,
                col=col,
            )
            fig.update_yaxes(
                tickvals=list(range(domain_samples.shape[0])),
                ticktext=domain_samples[:, 0],
                title="x1",
                row=1,
                col=col,
            )

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            width=1600,
            height=800,
            autosize=False,
        )
    else:
        raise ValueError(f"Unsupported number of input features: {model.n_input_feat}")

    return {"model": fig}


def visualize_input_data(
    model: Model,
    data_loader: DataLoader,
    scalers: List,
    noise_params: Dict,
):
    domain_samples = data_loader.dataset.tensors[0].numpy()
    fourier_series = data_loader.dataset.tensors[1].numpy().flatten()

    # create a 2D scatter plot that shows x1 x2 in x
    fig = px.scatter(
        x=domain_samples[:, 0],
        y=domain_samples[:, 1],
        color=fourier_series,
        marginal_x="box",
        marginal_y="box",
        title="Input & Target Data",
        opacity=0.3,
    )
    fig.update_layout(
        title_text=f"Input Data",
        plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_white",
        xaxis=dict(
            title="X2",
            showgrid=True,
        ),
        yaxis=dict(
            title="X1",
            showgrid=True,
        ),
        width=800,
        height=800,
    )

    return fig


def visualize_distribution(
    model: Model,
    data_loader: DataLoader,
    scalers: List,
    noise_params: Dict,
) -> go.Figure:
    # access dataset directly, we don't need batches now
    domain_samples = data_loader.dataset.tensors[0].numpy()
    fourier_series = data_loader.dataset.tensors[1].numpy()

    if type(model) == Model:
        prediction = model(
            params=model.params,
            inputs=domain_samples,
            noise_params=noise_params,
            execution_type="expval",
            force_mean=True,
        )
    else:
        prediction = model(torch.Tensor(domain_samples)).detach().numpy()

    # scaler 1 is for jets
    fourier_series = (
        scalers[1].inverse_transform(fourier_series.reshape(-1, 1)).flatten()
    )
    prediction = scalers[1].inverse_transform(prediction.reshape(-1, 1)).flatten()
    differences = prediction - fourier_series

    mlflow.log_metric("Differences Mean", differences.mean())
    mlflow.log_metric("Differences Std", differences.std())

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Distributions", "Differences"),
    )
    dists_plots = ff.create_distplot(
        [fourier_series, prediction, differences],
        ["Ground Truth", "Prediction", "Differences"],
        bin_size=2,
        curve_type="kde",
        show_rug=False,
    ).data

    fig.add_trace(
        dists_plots[0],
        row=1,
        col=1,
    )
    fig.add_trace(
        dists_plots[1],
        row=1,
        col=1,
    )
    fig.add_trace(
        dists_plots[3],
        row=1,
        col=1,
    )
    fig.add_trace(
        dists_plots[4],
        row=1,
        col=1,
    )
    fig.add_trace(
        dists_plots[2],
        row=2,
        col=1,
    )
    fig.add_trace(
        dists_plots[5],
        row=2,
        col=1,
    )

    fig.update_layout(
        title_text=f"Ground Truth and Model Prediction and Differences",
        plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_white",
        xaxis=dict(
            title="pt",
            showgrid=False,
        ),
        xaxis2=dict(
            title="pt_diff",
            showgrid=False,
        ),
        yaxis=dict(
            showgrid=False,
        ),
    )

    return {"model": fig}
