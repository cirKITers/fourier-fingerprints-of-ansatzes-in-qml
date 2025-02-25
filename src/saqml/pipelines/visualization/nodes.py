from qml_essentials.model import Model

import pennylane.numpy as pnp
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import mlflow
import logging

from typing import Dict

log = logging.getLogger(__name__)


def visualize_heatmap_filtered(
    df: pd.DataFrame,
) -> go.Figure:

    fig = go.Figure(
        data=go.Heatmap(
            z=df,
            y=df.index,
            x=df.columns,
            hoverongaps=False,
            colorscale="Sunset",
            zmax=1.0,
            zmin=0.0,
        )
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        width=600,
        height=600,
        autosize=False,
    )
    return fig


def visualize_coefficients_decay(df: pd.DataFrame) -> go.Figure:
    for i, c in enumerate(df):
        mlflow.log_metric(f"coefficients_decay", c, step=i)

    fig = go.Figure(data=go.Scatter(x=df.index, y=df, mode="markers+lines"))
    fig.update_layout(
        template="plotly_white",
        title="Coefficient Mean",
        yaxis_type="log",
    )

    return fig


def visualize_coefficients_correlated(
    df: pd.DataFrame, model: Model, discard_negative=True, triu=False
) -> go.Figure:
    if discard_negative:
        df_filtered = df.filter(regex="c_\+?\d+", axis=0).filter(
            regex="c_\+?\d+", axis=1
        )
    else:
        df_filtered = df.filter(regex="c.*", axis=0).filter(regex="c.*", axis=1)

    # nc = df_filtered.shape[0]
    # weights = np.flip(np.mgrid[0:nc:1, 0:nc:1].sum(axis=0) / ((nc - 1) * 2))
    # np.fill_diagonal(weights, 1)
    # df_filtered_weighted = df_filtered * weights

    if triu:
        for i in range(df_filtered.shape[0]):
            for j in range(df_filtered.shape[1]):
                if i <= j:
                    df_filtered.iloc[i, j] = pnp.nan
                    # df_filtered_weighted.iloc[i, j] = pnp.nan
        df_filtered = df_filtered.dropna(how="all", axis=0).dropna(how="all", axis=1)
        # df_filtered_weighted = df_filtered_weighted.dropna(how="all", axis=0).dropna(
        #     how="all", axis=1
        # )

    # mlflow.log_metric(
    #     "coefficients_correlation_weighted_variance", df_filtered_weighted.var().var()
    # )
    # mlflow.log_metric(
    #     "coefficients_correlation_weighted_mean", df_filtered_weighted.mean().mean()
    # )

    mlflow.log_metric("coefficients_correlation_variance", df_filtered.var().var())
    mlflow.log_metric("coefficients_correlation_mean", df_filtered.mean().mean())
    mlflow.log_metric("coefficients_correlation_max", df_filtered.max().max())
    mlflow.log_metric("coefficients_correlation_min", df_filtered.min().min())

    fig = visualize_heatmap_filtered(
        df=df_filtered,
    )

    fig.update_layout(
        title_text=f"Correlated Coefficients for {model.pqc.__class__.__name__}",
        xaxis=dict(
            title="Coefficients",
        ),
        yaxis=dict(title="Coefficients", autorange="reversed", scaleanchor="x"),
    )

    return fig


def visualize_parameters_correlated(
    df: pd.DataFrame, model: Model, triu=False
) -> go.Figure:
    df_filtered = df.filter(regex="p.*", axis=0).filter(regex="p.*", axis=1)

    if triu:
        for i in range(df_filtered.shape[0]):
            for j in range(df_filtered.shape[1]):
                if i <= j:
                    df_filtered.iloc[i, j] = pnp.nan
        df_filtered = df_filtered.dropna(how="all", axis=0).dropna(how="all", axis=1)

    mlflow.log_metric("parameters_correlation_variance", df_filtered.var().var())
    mlflow.log_metric("parameters_correlation_mean", df_filtered.mean().mean())
    mlflow.log_metric("parameters_correlation_max", df_filtered.max().max())
    mlflow.log_metric("parameters_correlation_min", df_filtered.min().min())

    fig = visualize_heatmap_filtered(
        df=df_filtered,
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
    if discard_negative:
        df_filtered = df.filter(regex="p_\d+", axis=0).filter(regex="c_\+?\d+", axis=1)
    else:
        df_filtered = df.filter(regex="p_\d+", axis=0).filter(regex="c.*", axis=1)

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
    domain_samples: pnp.ndarray,
    fourier_series: pnp.ndarray,
    noise_params: Dict,
) -> go.Figure:
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
