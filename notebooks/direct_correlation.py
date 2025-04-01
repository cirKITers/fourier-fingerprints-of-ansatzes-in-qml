import plotly
from plotly.subplots import make_subplots
import plotly.io as pio
import pandas as pd
import plotly.graph_objects as go
from runs.training_runs import run_ids as training_run_ids
from runs.training_runs import experiment_id
from runs.coefficient_runs import run_ids as coefficient_run_ids
from runs.expr_runs import run_ids as expr_run_ids
from helper import (
    save_fig,
    get_training_df,
    get_coefficient_df,
    get_expressibility_df,
    assign_ansatz_id,
    rgb_to_rgba,
    get_symbol_iterator,
    get_color_iterator,
)


pio.kaleido.scope.mathjax = None

cutoff_steps = 1e-3
weighted = True

training_df = get_training_df(training_run_ids, cutoff_steps=cutoff_steps)

coefficients_df = get_coefficient_df(coefficient_run_ids)

expr_df = get_expressibility_df(expr_run_ids)

coeff_expr_df = pd.merge(
    coefficients_df,
    expr_df,
    on=["ansatz", "qubits", "seed"],
)

combined_df = pd.merge(
    training_df,
    coeff_expr_df,
    on=["ansatz", "qubits", "seed"],
)
combined_df = assign_ansatz_id(combined_df)
# combined_df.sort_values(by="ansatz_id", inplace=True)

qubits = combined_df.qubits.unique()
ansaetze = combined_df.ansatz.unique()
ansatz_ids = combined_df.ansatz_id.unique()
seeds = combined_df.seed.unique()

pca_dataset = pd.DataFrame(
    columns=[
        "qubits",
        "ansatz",
        "ansatz_id",
        "corr_mean",
        "corr_max",
        "corr_min",
        "corr_var",
        "kl_divergence",
        "steps",
        "steps_var",
        "mse_min",
        "mse_min_var",
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
            pca_dataset.loc[idx, "ansatz_id"] = int(current_dataset.ansatz_id.mean())
            pca_dataset.loc[idx, "ansatz"] = ansaetze[pca_dataset.loc[idx, "ansatz_id"]]
            # pca_dataset.loc[idx, "coefficient_correlation"] = np.array(sub_fig_trace.z)
            pca_dataset.loc[idx, "corr_mean"] = (
                current_dataset.coefficients_correlation_mean.mean()
            )
            pca_dataset.loc[idx, "corr_w_mean"] = (
                current_dataset.coefficients_correlation_weighted_mean.mean()
            )
            pca_dataset.loc[idx, "corr_max"] = (
                current_dataset.coefficients_correlation_max.mean()
            )
            pca_dataset.loc[idx, "corr_min"] = (
                current_dataset.coefficients_correlation_min.mean()
            )
            pca_dataset.loc[idx, "corr_var"] = (
                current_dataset.coefficients_correlation_variance.mean()
            )
            pca_dataset.loc[idx, "steps"] = current_dataset.steps.mean()
            pca_dataset.loc[idx, "steps_var"] = current_dataset.steps.var()
            pca_dataset.loc[idx, "mse_min"] = current_dataset.mse_min.mean()
            pca_dataset.loc[idx, "mse_min_var"] = current_dataset.mse_min.var()
            pca_dataset.loc[idx, "kl_divergence"] = current_dataset.kl_divergence.mean()

            idx += 1

corr_mean = "corr_mean" if not weighted else "corr_w_mean"

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
                    ][corr_mean].mean()
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
                        ][corr_mean].std()
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
            title="Correlation Mean" if not weighted else "Weighted Correlation Mean",
        ),
        yaxis=dict(
            title=metric.title(),
            showgrid=False,
        ),
        # xaxis_type="log",
        # yaxis_type="log",
        showlegend=True,
    )

    save_fig(
        fig,
        f"direct_correlation_{metric}_c{cutoff_steps}_weighted_l",
        coefficient_run_ids + training_run_ids,
        experiment_id,
    )
