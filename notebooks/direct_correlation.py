import plotly
from plotly.subplots import make_subplots
import plotly.io as pio
import pandas as pd
import plotly.graph_objects as go
from runs.training_runs import run_ids as training_run_ids
from runs.training_runs import experiment_id
from runs.coefficient_runs import run_ids as coefficient_run_ids
from helper import (
    get_coefficient_df,
    get_training_df,
    assign_ansatz_id,
    get_symbol_iterator,
    get_color_iterator,
    save_fig,
)


pio.kaleido.scope.mathjax = None

training_df = get_training_df(training_run_ids)
training_df = training_df.rename(
    columns={
        "run_id": "training_run_id",
    }
)
coefficients_df = get_coefficient_df(coefficient_run_ids)
coefficients_df = coefficients_df.rename(
    columns={
        "run_id": "coefficient_run_id",
    }
)

combined_df = pd.merge(training_df, coefficients_df, on=["ansatz", "qubits", "seed"])
combined_df = assign_ansatz_id(combined_df)
combined_df.sort_values(by="ansatz_id", inplace=True)

qubits = combined_df.qubits.unique()
ansaetze = combined_df.ansatz.unique()
ansatz_ids = combined_df.ansatz_id.unique()
seeds = combined_df.seed.unique()

pca_dataset = pd.DataFrame(
    columns=[
        "qubits",
        "ansatz_id",
        "corr_mean",
        "corr_max",
        "corr_min",
        "corr_var",
        "steps",
    ]
)
idx = 0
for q in qubits:
    for seed in seeds:
        for it, ansatz in enumerate(ansaetze):
            current_dataset = combined_df[
                (combined_df.qubits == q)
                & (combined_df.ansatz == ansatz)
                & (combined_df.seed == seed)
            ]
            # sub_fig_trace = get_plotly_artifact(
            #     combined_df[
            #         (combined_df.qubits == q)
            #         & (combined_df.ansatz == ansatz)
            #         & (combined_df.seed == seed)
            #     ].coefficient_run_id.item()
            # )
            pca_dataset.loc[idx, "qubits"] = q
            pca_dataset.loc[idx, "ansatz_id"] = current_dataset.ansatz_id.item()
            # pca_dataset.loc[idx, "coefficient_correlation"] = np.array(sub_fig_trace.z)
            pca_dataset.loc[idx, "corr_mean"] = (
                current_dataset.coefficients_correlation_mean.item()
            )
            pca_dataset.loc[idx, "corr_max"] = (
                current_dataset.coefficients_correlation_max.item()
            )
            pca_dataset.loc[idx, "corr_min"] = (
                current_dataset.coefficients_correlation_min.item()
            )
            pca_dataset.loc[idx, "corr_var"] = (
                current_dataset.coefficients_correlation_variance.item()
            )
            pca_dataset.loc[idx, "steps"] = current_dataset.steps.item()
            pca_dataset.loc[idx, "mse_min"] = current_dataset.mse_min.item()

            idx += 1

        pass

for metric in ["steps", "mse_min"]:
    fig = go.Figure()
    symbols = get_symbol_iterator()
    for q in qubits:
        main_colors_it, _ = get_color_iterator()
        symbol = next(symbols)
        for ansatz_id in ansatz_ids:
            fig.add_scatter(
                x=[
                    pca_dataset[
                        (pca_dataset.ansatz_id == ansatz_id) & (pca_dataset.qubits == q)
                    ].corr_mean.mean()
                ],
                y=[
                    pca_dataset[
                        (pca_dataset.ansatz_id == ansatz_id) & (pca_dataset.qubits == q)
                    ][metric].mean()
                ],
                error_x=dict(
                    type="data",
                    array=[
                        pca_dataset[
                            (pca_dataset.ansatz_id == ansatz_id)
                            & (pca_dataset.qubits == q)
                        ].corr_mean.std()
                    ],
                    visible=True,
                ),
                error_y=dict(
                    type="data",
                    array=[
                        pca_dataset[
                            (pca_dataset.ansatz_id == ansatz_id)
                            & (pca_dataset.qubits == q)
                        ][metric].std()
                    ],
                    visible=True,
                ),
                mode="markers",
                name=f"{ansaetze[ansatz_id]}, {q} Qubits",
                marker=dict(color=next(main_colors_it), symbol=symbol),
            )

    fig.update_layout(
        title_text="Direct Correlation",
        template="plotly_white",
        xaxis=dict(
            title="Correlation Mean",
        ),
        yaxis=dict(
            title=metric.title(),
            showgrid=False,
        ),
        xaxis_type="log",
        yaxis_type="log",
        showlegend=True,
    )

    save_fig(
        fig,
        f"direct_correlation_{metric}",
        coefficient_run_ids + training_run_ids,
        experiment_id,
    )
