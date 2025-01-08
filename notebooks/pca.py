from plotly.subplots import make_subplots
import plotly.io as pio
import pandas as pd
from runs.training_runs import run_ids as training_run_ids
from runs.training_runs import experiment_id
from runs.coefficient_runs import run_ids as coefficient_run_ids
from helper import (
    get_coefficient_df,
    get_training_df,
    assign_ansatz_id,
    generate_hash,
    get_plotly_artifact,
)
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

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
combined_df.sort_values(by="qubits", inplace=True)
combined_df = assign_ansatz_id(combined_df)

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
        fig = make_subplots(rows=1, cols=len(ansaetze), subplot_titles=ansaetze)

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

            idx += 1

        pass

sel_pca_dataset = pca_dataset.drop(
    columns=["qubits", "ansatz_id", "corr_max", "corr_min", "corr_var"]
)
pca_dataset_scaled = (sel_pca_dataset - sel_pca_dataset.mean()) / sel_pca_dataset.std()
coeff_pca = PCA(n_components=2).fit(pca_dataset_scaled)
print(
    f"Proportion of variance explained by each principal component:\n{coeff_pca.explained_variance_ratio_}"
)

scores = coeff_pca.transform(pca_dataset_scaled)
# biplot
fig, ax = plt.subplots(figsize=(8, 8))
colormap = plt.cm.Dark2
all_ansatz_ids = pca_dataset.ansatz_id.to_list()
ansatz_colors = [colormap.colors[i] for i in all_ansatz_ids]
ax.scatter(
    scores[:, 0],
    scores[:, 1],
    color=ansatz_colors,  # color based on the ansatz
    # edgecolor=ansatz_colors,
    # alpha=0.5,
)

prop = dict(arrowstyle="-|>,head_width=0.4,head_length=0.8", shrinkA=0, shrinkB=0)
for i in range(coeff_pca.components_.shape[1]):
    length = np.sqrt(
        coeff_pca.components_[0, i] ** 2 + coeff_pca.components_[1, i] ** 2
    )
    angle = np.arctan2(coeff_pca.components_[1, i], coeff_pca.components_[0, i])
    ax.annotate(
        "",
        xy=(coeff_pca.components_[0, i], coeff_pca.components_[1, i]),
        xytext=(0, 0),
        # head_width=0.1,
        # head_length=0.1,
        # linewidth=2,
        # color="red",
        arrowprops=prop,
    )
    ax.text(
        (length + 0.2 + 0.3 * np.cos(angle)) * np.cos(angle),
        (length + 0.2 + 0.3 * np.cos(angle)) * np.sin(angle),
        pca_dataset_scaled.columns[i],
        # color="black",
        ha="center",
        va="center",
    )
# for i in range(scores.shape[0]):
#     ax.text(
#         scores[i, 0] + 0.2,
#         scores[i, 1],
#         pca_dataset_scaled.index[i],
#         color="blue",
#         ha="center",
#         va="center",
#     )
ax.set_xlabel(
    f"PC1 ({coeff_pca.explained_variance_ratio_[1] * 100:.1f}% Variance Ratio)"
)
ax.set_ylabel(
    f"PC2 ({coeff_pca.explained_variance_ratio_[0] * 100:.1f}% Variance Ratio)"
)
ax.set_title("Biplot - Qubits, # of Steps, Ansatz and Correlation Mean")
hs = generate_hash(coefficient_run_ids + training_run_ids)
name = "pca"
path = f"results/{experiment_id}/{hs}/"
print(f"Saving figure to {path}{name}.pdf")
plt.savefig(f"{path}{name}.pdf", format="pdf", bbox_inches="tight")
