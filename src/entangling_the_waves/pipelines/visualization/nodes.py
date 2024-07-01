from qml_essentials.model import Model

import plotly.graph_objects as go
import pandas as pd
import mlflow
import logging

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
