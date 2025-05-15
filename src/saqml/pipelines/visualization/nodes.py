from qml_essentials.model import Model
from torch.utils.data import DataLoader
import torch
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pennylane.numpy as pnp
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
import pandas as pd
import mlflow
import plotly.express as px
import logging

from typing import Dict, List

log = logging.getLogger(__name__)


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
        width=600,
        height=600,
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
        fig = visualize_heatmap_filtered(df=heatmap, zmax=heatmap.max().max())
    else:
        raise NotImplementedError(
            "Only implemented for n_input_feat=1 and n_input_feat=2"
        )

    return fig


def visualize_spectrum(df: pd.DataFrame, model: Model) -> go.Figure:
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
        fig.update_layout(template="plotly_white", title="Spectrum", showlegend=False)
    elif model.n_input_feat == 2:
        df_mean = df.mean(axis=1)
        fig = go.Figure()
    else:
        raise NotImplementedError(
            "Only implemented for n_input_feat=1 and n_input_feat=2"
        )

    return fig


def visualize_coefficients_correlated(
    df: pd.DataFrame, model: Model, discard_negative=True, triu=False
) -> go.Figure:

    if triu:
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                if i <= j:
                    df.iloc[i, j] = pnp.nan
                    # df_weighted.iloc[i, j] = pnp.nan
        df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
        # df_weighted = df_weighted.dropna(how="all", axis=0).dropna(
        #     how="all", axis=1
        # )

    mlflow.log_metric("coefficients_correlation_variance", df.var().var())
    mlflow.log_metric("coefficients_correlation_mean", df.mean().mean())
    mlflow.log_metric("coefficients_correlation_max", df.max().max())
    mlflow.log_metric("coefficients_correlation_min", df.min().min())

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

    return fig


def visualize_coefficients_correlated_filtered_weighted(
    df: pd.DataFrame, model: Model, discard_negative=True, triu=False
) -> go.Figure:

    if triu:
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                if i <= j:
                    df.iloc[i, j] = pnp.nan
                    # df_weighted.iloc[i, j] = pnp.nan
        df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
        # df_weighted = df_weighted.dropna(how="all", axis=0).dropna(
        #     how="all", axis=1
        # )

    mlflow.log_metric("coefficients_correlation_weighted_variance", df.var().var())
    mlflow.log_metric("coefficients_correlation_weighted_mean", df.mean().mean())
    mlflow.log_metric("coefficients_correlation_weighted_max", df.max().max())
    mlflow.log_metric("coefficients_correlation_weighted_min", df.min().min())

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
                    df.iloc[i, j] = pnp.nan
        df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)

    mlflow.log_metric("parameters_correlation_variance", df.var().var())
    mlflow.log_metric("parameters_correlation_mean", df.mean().mean())
    mlflow.log_metric("parameters_correlation_max", df.max().max())
    mlflow.log_metric("parameters_correlation_min", df.min().min())

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
    df_filtered = df.filter(regex="p_\d+", axis=0).filter(
        regex=f"c(_\+\d+){{{model.n_input_feat}}}", axis=1
    )

    mlflow.log_metric(
        "parameters_coefficients_correlation_variance", df_filtered.var().var()
    )
    mlflow.log_metric(
        "parameters_coefficients_correlation_mean", df_filtered.mean().mean()
    )
    mlflow.log_metric(
        "parameters_coefficients_correlation_max", df_filtered.max().max()
    )
    mlflow.log_metric(
        "parameters_coefficients_correlation_min", df_filtered.min().min()
    )

    fig = visualize_heatmap_filtered(
        df=df_filtered,
    )
    fig.update_layout(
        title_text=f"Correlation Parameters Coefficients for {model.pqc.__class__.__name__}",
        xaxis=dict(
            title="Coefficients",
        ),
        yaxis=dict(title="Parameters", autorange="reversed", scaleanchor="x"),
    )

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

    fig = go.Figure(
        data=[
            go.Scatter(
                x=domain_samples, y=fourier_series, mode="lines", name="Ground Truth"
            ),
            go.Scatter(
                x=domain_samples,
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
            title="Input Value",
            showgrid=False,
        ),
        yaxis=dict(
            # title="Correlation Mean",
            showgrid=False,
        ),
    )

    return {"model": fig}


def visualize_data(
    model: Model,
    data_loader: DataLoader,
    scalers: List,
    noise_params: Dict,
) -> go.Figure:
    # access dataset directly, we don't need batches now
    domain_samples = data_loader.dataset.tensors[0].numpy()
    fourier_series = data_loader.dataset.tensors[1].numpy()
    prediction = model(
        params=model.params,
        inputs=domain_samples,
        noise_params=noise_params,
        execution_type="expval",
        force_mean=True,
    )

    # scaler 1 is for jets
    fourier_series = (
        scalers[1].inverse_transform(fourier_series.reshape(-1, 1)).flatten()
    )
    prediction = scalers[1].inverse_transform(prediction.reshape(-1, 1)).flatten()

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Distributions", "Differences"),
    )
    dists_plots = ff.create_distplot(
        [fourier_series, prediction, prediction - fourier_series],
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
