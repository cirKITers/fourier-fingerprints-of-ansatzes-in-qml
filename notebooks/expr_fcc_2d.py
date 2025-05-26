import pandas as pd

from helper import (
    save_fig,
    get_run_ids,
    get_training_df,
    get_coefficient_df,
    get_expressibility_df,
    assign_ansatz_id,
    visualize_boxplot,
)

unique_id = "expr_fcc"

scenarios = {
    "1DFS": {
        "training_experiment_id": "499640227395518059",
        "coefficient_id": "258301106012425434",  # change!
        "expr_id": "855134941797278912",
        "metric": "mse_valid",
        "cutoff_steps": 1e-2,
    },
    "2DFS": {
        "training_experiment_id": "964165187008575029",
        "coefficient_id": "258301106012425434",
        "expr_id": "855134941797278912",
        "metric": "mse_valid",
        "cutoff_steps": 1e-2,
    },
    "2DHEP": {
        "training_experiment_id": "240205035422235647",
        "coefficient_id": "258301106012425434",
        "expr_id": "855134941797278912",
        "metric": "mse_valid",
        "cutoff_steps": 1e-2,
    },
}
for scenario, setting in scenarios.items():
    metric = setting["metric"]
    cutoff_steps = setting["cutoff_steps"]

    # get run_ids
    training_run_ids = get_run_ids(setting["training_experiment_id"])
    coefficient_run_ids = get_run_ids(setting["coefficient_id"])
    expr_run_ids = get_run_ids(setting["expr_id"])

    # get dataframes
    training_df = get_training_df(
        training_run_ids, cutoff_steps=cutoff_steps, metric=metric
    )
    coefficients_df = get_coefficient_df(coefficient_run_ids)
    expr_df = get_expressibility_df(expr_run_ids)

    # combine dataframes
    combined_df = pd.merge(
        training_df,
        pd.merge(
            coefficients_df,
            expr_df,
            on=["ansatz", "qubits", "seed"],
        ),
        on=["ansatz", "qubits", "seed"],
    )
    # add ids to ansatz
    combined_df = assign_ansatz_id(combined_df)

    # get unique values
    qubits = combined_df.qubits.unique()
    ansaetze = combined_df.ansatz.unique()
    ansatz_ids = combined_df.ansatz_id.unique()
    seeds = combined_df.seed.unique()

    # build new "clean" dataframe
    df = pd.DataFrame(
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
    for q in sorted(qubits):
        for seed in sorted(seeds):
            for it, ansatz in enumerate(ansaetze):
                current_dataset = combined_df[
                    (combined_df.qubits == q)
                    & (combined_df.ansatz == ansatz)
                    & (combined_df.seed == seed)
                ]
                if len(current_dataset) == 0:
                    print(f"No data for q={q}, ansatz={ansatz}, seed={seed}")
                    continue

                df.loc[idx, "qubits"] = q
                df.loc[idx, "ansatz_id"] = int(current_dataset.ansatz_id.mean())
                df.loc[idx, "ansatz"] = ansatz
                # pca_dataset.loc[idx, "coefficient_correlation"] = np.array(sub_fig_trace.z)
                df.loc[idx, "corr_mean"] = (
                    current_dataset.coefficients_correlation_mean.mean()
                )
                df.loc[idx, "corr_w_mean"] = (
                    current_dataset.coefficients_correlation_weighted_mean.mean()
                )
                df.loc[idx, "corr_max"] = (
                    current_dataset.coefficients_correlation_max.mean()
                )
                df.loc[idx, "corr_min"] = (
                    current_dataset.coefficients_correlation_min.mean()
                )
                df.loc[idx, "corr_var"] = (
                    current_dataset.coefficients_correlation_variance.mean()
                )
                df.loc[idx, "steps"] = current_dataset.steps.mean()
                df.loc[idx, "steps_var"] = current_dataset.steps.var()
                df.loc[idx, f"{metric}_min"] = current_dataset[f"{metric}_min"].mean()
                df.loc[idx, f"{metric}_min_var"] = current_dataset[
                    f"{metric}_min"
                ].var()
                df.loc[idx, "kl_divergence"] = current_dataset.kl_divergence.mean()

                idx += 1

    for metric in ["steps", f"{metric}_min"]:
        sorted_pca_dataset = df.sort_values(by=metric, ascending=False)
        for q in qubits:
            fig = visualize_boxplot(df, q, metric)

            save_fig(
                fig,
                f"{unique_id}_{metric}_c{cutoff_steps}_q{q}",
                expr_run_ids + coefficient_run_ids + training_run_ids,
                unique_id,
            )
