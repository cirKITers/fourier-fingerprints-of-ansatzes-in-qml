import plotly
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio
from plotly.subplots import make_subplots

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
            pca_dataset.loc[idx, "mse_min"] = current_dataset.mse_min.mean()
            pca_dataset.loc[idx, "kl_divergence"] = current_dataset.kl_divergence.mean()

            idx += 1


for metric in ["steps", "mse_min"]:
    sorted_pca_dataset = pca_dataset.sort_values(by=metric, ascending=False)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    symbols = get_symbol_iterator()
    for q in qubits:
        symbol = next(symbols)
        main_colors_it, _ = get_color_iterator()
        fig.add_trace(
            go.Scatter(
                x=sorted_pca_dataset.ansatz,
                y=sorted_pca_dataset.corr_mean,
                name=f"Corr. Mean",
                mode="markers",
                marker=dict(color=next(main_colors_it), symbol=symbol),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=sorted_pca_dataset.ansatz,
                y=sorted_pca_dataset.corr_w_mean,
                name=f"Corr. Weight. Mean",
                mode="markers",
                marker=dict(color=next(main_colors_it), symbol=symbol),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=sorted_pca_dataset.ansatz,
                y=sorted_pca_dataset.kl_divergence,
                name=f"KL Divergence",
                mode="markers",
                marker=dict(color=next(main_colors_it), symbol=symbol),
            ),
            secondary_y=True,
        )

        fig.add_annotation(
            dict(
                x=len(ansaetze),
                y=1.5,
                xref="x",
                yref="y",
                ax=2,
                ay=1.5,
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="gray",
                text="Low MSE",
            )
        )

        fig.update_yaxes(title_text=f"Correlation", secondary_y=False)
        fig.update_yaxes(title_text="KL Divergence", secondary_y=True)
        fig.update_layout(
            title=f"Correlation and Expressibility ({q} Qubits)",
            template="plotly_white",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            # margin=dict(l=150, r=30, b=30),
        )

        save_fig(
            fig,
            f"cor_expr_{metric}_c{cutoff_steps}_q{q}",
            coefficient_run_ids + training_run_ids,
            experiment_id,
        )
