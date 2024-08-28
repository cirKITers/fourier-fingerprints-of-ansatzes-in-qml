from qml_essentials.model import Model

import pennylane.numpy as np
import plotly.graph_objects as go
import pandas as pd
import mlflow
import logging

from typing import Dict

log = logging.getLogger(__name__)


def visualize_heatmap_filtered(
    df: pd.DataFrame,
    title: str,
    name: str,
) -> go.Figure:

    fig = go.Figure(
        data=go.Heatmap(
            z=df,
            y=df.index,
            x=df.columns,
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title_text=title,
        plot_bgcolor="rgba(0,0,0,0)",
        width=600,
        height=600,
        autosize=False,
        xaxis=dict(
            title="Coefficients",
        ),
        yaxis=dict(title="Coefficients", autorange="reversed", scaleanchor="x"),
    )
    mlflow.log_figure(fig, f"{name}.html")
    return fig


def visualize_coefficients_correlated(
    df: pd.DataFrame,
    model: Model,
) -> go.Figure:
    df_filtered = df.filter(regex="c.*", axis=0).filter(regex="c.*", axis=1)
    mlflow.log_metric("coefficients_correlation_variance", df_filtered.var().var())
    mlflow.log_metric("coefficients_correlation_mean", df_filtered.mean().mean())
    return visualize_heatmap_filtered(
        df=df_filtered,
        title=f"Correlated Coefficients for {model.pqc.__class__.__name__}",
        name=f"correlated_coefficients_{model.pqc.__class__.__name__.lower()}",
    )


def visualize_parameters_correlated(
    df: pd.DataFrame,
    model: Model,
) -> go.Figure:
    df_filtered = df.filter(regex="p.*", axis=0).filter(regex="p.*", axis=1)
    mlflow.log_metric("parameters_correlation_variance", df_filtered.var().var())
    mlflow.log_metric("parameters_correlation_mean", df_filtered.mean().mean())
    return visualize_heatmap_filtered(
        df=df_filtered,
        title=f"Correlated Parameters for {model.pqc.__class__.__name__}",
        name=f"correlated_parameters_{model.pqc.__class__.__name__.lower()}",
    )


def visualize_parameters_coefficients_correlated(
    df: pd.DataFrame,
    model: Model,
) -> go.Figure:
    df_filtered = df.filter(regex="p.*", axis=0).filter(regex="c.*", axis=1)
    mlflow.log_metric(
        "parameters_coefficients_correlation_variance", df_filtered.var().var()
    )
    mlflow.log_metric(
        "parameters_coefficients_correlation_mean", df_filtered.mean().mean()
    )
    return visualize_heatmap_filtered(
        df=df_filtered,
        title=f"Correlation Parameters Coefficients for {model.pqc.__class__.__name__}",
        name=f"correlated_parameters_coefficients_{model.pqc.__class__.__name__.lower()}",
    )


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
    mlflow.log_figure(fig, f"coefficients_correlated_control.html")
    return fig


def visualize_model(
    model: Model,
    domain_samples: np.ndarray,
    fourier_series: np.ndarray,
    noise_params: Dict,
) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Scatter(
                x=domain_samples, y=fourier_series, mode="lines", name="Fourier Series"
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
                name="Ground Truth",
            ),
        ]
    )
    fig.update_layout(
        title_text=f"Ground Truth and Model Prediction",
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
    mlflow.log_figure(fig, f"coefficients_correlated_control.html")

    return fig
