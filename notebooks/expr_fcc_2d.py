import pandas as pd

from helper import (
    save_fig,
    cache_df,
    get_run_ids,
    get_training_df,
    get_coefficient_df,
    get_expressibility_df,
    assign_ansatz_id,
    visualize_boxplot,
    visualize_scatter,
)

cache = True
unique_id = "expr_fcc"

scenarios = {
    "1DFS": {
        "training_experiment_id": "499640227395518059",
        "coefficient_id": "348625906624370813",
        "expr_id": "198634010954150003",
        "metric": "mse_valid",
        "cutoff_steps": 1e-2,
    },
    "2DFS": {
        "training_experiment_id": "964165187008575029",
        "coefficient_id": "219295129676641074",
        "expr_id": "198634010954150003",
        "metric": "mse_valid",
        "cutoff_steps": 1e-2,
    },
    "2DHEP": {
        "training_experiment_id": "240205035422235647",
        "coefficient_id": "219295129676641074",
        "expr_id": "198634010954150003",
        "metric": "mse_valid",
        "cutoff_steps": 1e-2,
    },
}
for scenario, setting in scenarios.items():
    print(f"{'-' * 100}")
    print(f"\nScenario: {scenario}\n")
    print(f"{'-' * 100}")

    metric = setting["metric"]
    cutoff_steps = setting["cutoff_steps"]

    # get run_ids
    training_run_ids = get_run_ids(setting["training_experiment_id"])
    coefficient_run_ids = get_run_ids(setting["coefficient_id"])
    expr_run_ids = get_run_ids(setting["expr_id"])

    df = cache_df(training_run_ids + coefficient_run_ids + expr_run_ids)
    if not cache or df is None:

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
                "coeff_run_id",
                "training_run_id",
                "expr_run_id",
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
                        _occurences = []
                        for _df in [training_df, coefficients_df, expr_df]:
                            _occurences.append(
                                len(
                                    _df[
                                        (_df.qubits == q)
                                        & (_df.ansatz == ansatz)
                                        & (_df.seed == seed)
                                    ]
                                )
                            )
                        print(
                            f"Occurences:\nTraining ({setting['training_experiment_id']}) {_occurences[0]}\nCoefficients ({setting['coefficient_id']}) {_occurences[1]}\nExpressibility ({setting['expr_id']}) {_occurences[2]}\n"
                        )
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
                    df.loc[idx, f"{metric}_min"] = current_dataset[
                        f"{metric}_min"
                    ].mean()
                    df.loc[idx, f"{metric}_min_var"] = current_dataset[
                        f"{metric}_min"
                    ].var()
                    df.loc[idx, "kl_divergence"] = current_dataset.kl_divergence.mean()

                    df.loc[idx, "coeff_run_id"] = (
                        current_dataset.coefficient_run_id.item()
                    )
                    # df.loc[idx, "training_run_id"] = (
                    #     current_dataset.training_run_id.item()
                    # )
                    df.loc[idx, "expr_run_id"] = current_dataset.expr_run_id.item()

                    idx += 1

        print(f"Caching dataframe: {df.describe()}")
        cache_df(run_ids=training_run_ids + coefficient_run_ids + expr_run_ids, df=df)
    else:
        print(f"Using cached dataframe: {df.describe()}")
        # get unique values
        qubits = df.qubits.unique()
        ansaetze = df.ansatz.unique()
        ansatz_ids = df.ansatz_id.unique()

    for metric in [f"{metric}_min"]:
        sorted_df = df.sort_values(by=metric, ascending=False)
        for q in qubits:
            fig = visualize_boxplot(sorted_df[sorted_df.qubits == q], metric)
            fig.update_layout(title=f"{scenario}, {q} Qubits, Metric: {metric}")
            save_fig(
                fig,
                f"{scenario}_{unique_id}_bp_{metric}_c{cutoff_steps}_q{q}",
                expr_run_ids + coefficient_run_ids + training_run_ids,
                unique_id,
            )

            fig = visualize_scatter(
                sorted_df[sorted_df.qubits == q],
                ansatz_ids,
                metric,
                weighted=weighted,
            )
            fig.update_layout(title=f"{scenario}, {q} Qubits, Metric: {metric})")
            save_fig(
                fig,
                f"{scenario}_{unique_id}_sc_{metric}_c{cutoff_steps}_q{q}_{'w' if weighted else 'uw'}",
                expr_run_ids + coefficient_run_ids + training_run_ids,
                unique_id,
            )

            fig = visualize_heatmap(sorted_df[sorted_df.qubits == q], 1000, weighted)
            save_fig(
                fig,
                f"{scenario}_{unique_id}_hm_{metric}_c{cutoff_steps}_q{q}_{'w' if weighted else 'uw'}",
                expr_run_ids + coefficient_run_ids + training_run_ids,
                unique_id,
            )
