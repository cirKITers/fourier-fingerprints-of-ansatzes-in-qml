import pandas as pd

from helper import (
    save_fig,
    cache_df,
    get_run_ids,
    get_training_df,
    get_classical_training_df,
    get_coefficient_df,
    get_expressibility_df,
    assign_ansatz_id,
    visualize_boxplot,
    visualize_scatter,
    visualize_heatmap,
    visualize_distribution,
    visualize_single_heatmap,
    visualize_expr_scatter,
)
import json

cache = True
weighted = False
unique_id = "expr_fcc"

scenarios = {
    # "1DFS": {
    #     "training_experiment_id": "499640227395518059",  # RX-enc: 499640227395518059, RY-enc: 264811618779563708
    #     "coefficient_id": "294759570659091329",  # RX-enc: 294759570659091329, RY-enc: 286271885992155758
    #     "expr_id": "182562157534908977",
    #     "metric": "mse_valid",
    #     "cutoff_steps": 1e-2,
    # },
    # "2DFS": {
    #     "training_experiment_id": "964165187008575029",
    #     "coefficient_id": "452677263305714256",
    #     "expr_id": "182562157534908977",
    #     "metric": "mse_valid",
    #     "cutoff_steps": 1e-2,
    # },
    "2DHEP": {
        "training_experiment_id": "100058640076220878",  # 3000 steps: 240205035422235647, 1000 steps: 547640067507594003 # dists: 100058640076220878
        "classical_training_experiment_id": "310042118257145976",
        "coefficient_id": "452677263305714256",
        "expr_id": "182562157534908977",
        "metric": "mse_valid",
        "cutoff_steps": 1e-2,
    },
}
enabled_plots = ["sce", "hm", "hms", "dist"]  # "bp", "sc", "sce", "hm", "hms"

missing_items = {"1DFS": [], "2DFS": [], "2DHEP": []}

for scenario, setting in scenarios.items():
    print(f"{'-' * 100}")
    print(f"\nScenario: {scenario}\n")
    print(f"{'-' * 100}")

    metric = setting["metric"]
    cutoff_steps = setting["cutoff_steps"]

    # get run_ids
    coefficient_run_ids = get_run_ids(setting["coefficient_id"])
    expr_run_ids = get_run_ids(setting["expr_id"])
    training_run_ids = get_run_ids(setting["training_experiment_id"])
    if "classical_training_experiment_id" in setting:
        classical_training_run_ids = get_run_ids(
            setting["classical_training_experiment_id"]
        )
        df = cache_df(
            training_run_ids
            + classical_training_run_ids
            + coefficient_run_ids
            + expr_run_ids
        )
    else:
        classical_training_run_ids = None
        df = cache_df(training_run_ids + coefficient_run_ids + expr_run_ids)

    if not cache or df is None:

        # get dataframes
        training_df = get_training_df(
            training_run_ids, cutoff_steps=cutoff_steps, metric=metric
        )
        classical_training_df = get_classical_training_df(
            classical_training_run_ids, cutoff_steps=cutoff_steps, metric=metric
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
        combined_df = combined_df.sort_values(by="ansatz", ascending=False)
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
                "seed",
                "coeff_run_id",
                "training_run_id",
                "classical_training_run_id",
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
                        missing_items[scenario].append(
                            {
                                "configuration": {
                                    "qubits": q,
                                    "ansatz": ansatz,
                                    "seed": seed,
                                },
                                "occurences": {
                                    "training": _occurences[0],
                                    "coefficients": _occurences[1],
                                    "expressibility": _occurences[2],
                                },
                            }
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

                    df.loc[idx, "seed"] = seed
                    df.loc[idx, "coeff_run_id"] = current_dataset.coeff_run_id.unique()[
                        0
                    ]
                    df.loc[idx, "training_run_id"] = (
                        f"{current_dataset.training_run_id.to_list()}"
                    )
                    if classical_training_df is not None:
                        df.loc[idx, "classical_training_run_id"] = (
                            f"{classical_training_df[classical_training_df.seed==seed].training_run_id.to_list()}"
                        )
                    df.loc[idx, "expr_run_id"] = current_dataset.expr_run_id.unique()[0]

                    idx += 1

        print(f"Caching dataframe: {df.describe()}")
        if classical_training_df is not None:
            cache_df(
                run_ids=training_run_ids
                + classical_training_run_ids
                + coefficient_run_ids
                + expr_run_ids,
                df=df,
            )
        else:
            cache_df(
                run_ids=training_run_ids + coefficient_run_ids + expr_run_ids, df=df
            )
    else:
        print(f"Using cached dataframe: {df.describe()}")
        # get unique values
        qubits = df.qubits.unique()
        ansaetze = df.ansatz.unique()
        ansatz_ids = df.ansatz_id.unique()

    for metric in [f"{metric}_min"]:
        for q in qubits:
            if "bp" in enabled_plots:
                fig = visualize_boxplot(df[df.qubits == q], metric)
                fig.update_layout(title=f"{scenario}, {q} Qubits, Metric: {metric}")
                save_fig(
                    fig,
                    f"{scenario}_bp_{metric}_c{cutoff_steps}_q{q}",
                    expr_run_ids + coefficient_run_ids + training_run_ids,
                    scenario,
                )

            if "sc" in enabled_plots:
                fig = visualize_scatter(
                    df[df.qubits == q],
                    ansatz_ids,
                    metric,
                    weighted=weighted,
                )
                fig.update_layout(title=f"{scenario}, {q} Qubits, Metric: {metric})")
                save_fig(
                    fig,
                    f"{scenario}_sc_{metric}_c{cutoff_steps}_q{q}_{'w' if weighted else 'uw'}",
                    expr_run_ids + coefficient_run_ids + training_run_ids,
                    scenario,
                )

            if "sce" in enabled_plots:
                # scatter plot
                fig = visualize_expr_scatter(
                    df[df.qubits == q],
                    ansatz_ids,
                    metric,
                    weighted=weighted,
                    legendonly=False,
                )
                fig.update_layout(title=f"{scenario}, {q} Qubits, Metric: {metric})")
                fig.update_layout(title=f"")
                save_fig(
                    fig,
                    f"{scenario}_sce_{metric}_c{cutoff_steps}_q{q}_{'w' if weighted else 'uw'}",
                    expr_run_ids + coefficient_run_ids + training_run_ids,
                    scenario,
                    showlegend=False,
                    font_size=20,
                )

                # legendonly
                fig = visualize_expr_scatter(
                    df[df.qubits == q],
                    ansatz_ids,
                    metric,
                    weighted=weighted,
                    legendonly=True,
                )
                fig.update_layout(title=f"{scenario}, {q} Qubits, Metric: {metric})")
                fig.update_layout(title=f"")
                save_fig(
                    fig,
                    f"{scenario}_sce_{metric}_c{cutoff_steps}_q{q}_{'w' if weighted else 'uw'}_legend",
                    expr_run_ids + coefficient_run_ids + training_run_ids,
                    scenario,
                    showlegend=True,
                )

            if scenario == "1DFS" and "hm" in enabled_plots:
                fig = visualize_heatmap(df[df.qubits == q], 1000, weighted)
                fig.update_layout(title=f"")
                save_fig(
                    fig,
                    f"{scenario}_hm_{metric}_c{cutoff_steps}_q{q}_{'w' if weighted else 'uw'}",
                    expr_run_ids + coefficient_run_ids + training_run_ids,
                    scenario,
                    font_size=20,
                )

            if scenario == "2DFS" and "hms" in enabled_plots:
                fig = visualize_single_heatmap(
                    df[df.qubits == q], 1000, "Hardware_Efficient", weighted
                )
                fig.update_layout(title=f"")
                save_fig(
                    fig,
                    f"{scenario}_hms_{metric}_c{cutoff_steps}_q{q}_{'w' if weighted else 'uw'}",
                    expr_run_ids + coefficient_run_ids + training_run_ids,
                    scenario,
                    font_size=20,
                )

            if scenario == "2DHEP" and "dist" in enabled_plots:
                fig = visualize_distribution(
                    df[df.qubits == q], identifier="fig_distribution_train"
                )
                fig.update_layout(title=f"")
                save_fig(
                    fig,
                    f"{scenario}_dist_train_q{q}",
                    expr_run_ids + coefficient_run_ids + training_run_ids,
                    scenario,
                    font_size=20,
                )

                fig = visualize_distribution(
                    df[df.qubits == q], identifier="fig_distribution_valid"
                )
                fig.update_layout(title=f"")
                save_fig(
                    fig,
                    f"{scenario}_dist_valid_q{q}",
                    expr_run_ids + coefficient_run_ids + training_run_ids,
                    scenario,
                    font_size=20,
                )

with open("missing_items.json", "w") as f:
    json.dump(missing_items, f)
