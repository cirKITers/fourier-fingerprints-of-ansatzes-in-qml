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
    visualize_heatmap,
    visualize_distribution,
    visualize_single_heatmap,
    visualize_expr_scatter,
    visualize_coeff_param_relation,
    visualize_coeff_variance,
)
import json

cache = True
weighted = False
unique_id = "expr_fcc"

scenarios = {
    # "1dfs_rc": {
    #     "training_experiment_id": None,  # RX-enc: 499640227395518059, RY-enc: 264811618779563708
    #     "coefficient_id": "552541178809486661",  # RX-enc: 294759570659091329, RY-enc: 286271885992155758, RY-enc-2:654703589739658185
    #     "expr_id": None,
    #     "metric": None,
    #     "metric_name": None,
    #     "cutoff_steps": None,
    # },
    "1dfs_rx": {
        "training_experiment_id": "499640227395518059",  # RX-enc: 499640227395518059, RY-enc: 264811618779563708
        "coefficient_id": "294759570659091329",  # RX-enc: 294759570659091329, RY-enc: 286271885992155758, RY-enc-2:654703589739658185
        "expr_id": "182562157534908977",
        "metric": "mse_valid",
        "metric_name": "Mean Squared Error",
        "cutoff_steps": 1e-2,
    },
    "1dfs_ry": {
        "training_experiment_id": "264811618779563708",  # RX-enc: 499640227395518059, RY-enc: 264811618779563708
        "coefficient_id": "654703589739658185",  # RX-enc: 294759570659091329, RY-enc: 286271885992155758, RY-enc-2:654703589739658185
        "expr_id": "182562157534908977",
        "metric": "mse_valid",
        "metric_name": "Mean Squared Error",
        "cutoff_steps": 1e-2,
    },
    # "2dfs": {
    #     "training_experiment_id": "964165187008575029",
    #     "coefficient_id": "452677263305714256",
    #     "expr_id": "182562157534908977",
    #     "metric": "mse_valid",
    #     "metric_name": "Mean Squared Error",
    #     "cutoff_steps": 1e-2,
    # },
    # "2dhep_mse": {
    #     "training_experiment_id": "547640067507594003",  # 3000 steps: 240205035422235647, 1000 steps: 547640067507594003 # dists: 100058640076220878
    #     "coefficient_id": "452677263305714256",
    #     "expr_id": "182562157534908977",
    #     "metric": "mse_valid",  # mse_valid, kl_divergence_valid, huber_loss_valid
    #     "metric_name": "Mean Squared Error",  # Mean Squared Error, KL Divergence, Huber Loss
    #     "cutoff_steps": 1e-2,
    # },
    # "2dhep_kl": {
    #     "training_experiment_id": "547640067507594003",  # 3000 steps: 240205035422235647, 1000 steps: 547640067507594003 # dists: 100058640076220878
    #     "coefficient_id": "452677263305714256",
    #     "expr_id": "182562157534908977",
    #     "metric": "kl_divergence_valid",  # mse_valid, kl_divergence_valid, huber_loss_valid
    #     "metric_name": "KL Divergence",  # Mean Squared Error, KL Divergence, Huber Loss
    #     "cutoff_steps": 1e-2,
    # },
    # "2dhep_hl": {
    #     "training_experiment_id": "547640067507594003",  # 3000 steps: 240205035422235647, 1000 steps: 547640067507594003 # dists: 100058640076220878
    #     "coefficient_id": "452677263305714256",
    #     "expr_id": "182562157534908977",
    #     "metric": "huber_loss_valid",  # mse_valid, kl_divergence_valid, huber_loss_valid
    #     "metric_name": "Huber Loss",  # Mean Squared Error, KL Divergence, Huber Loss
    #     "cutoff_steps": 1e-2,
    # },
    # "2dhepc": {
    #     "training_experiment_id": "240205035422235647",  # 3000 steps: 240205035422235647, 1000 steps: 547640067507594003 # dists: 100058640076220878
    #     "classical_training_experiment_id": "310042118257145976",
    #     "coefficient_id": "452677263305714256",
    #     "expr_id": "182562157534908977",
    #     "metric": "mse_valid",  # mse_valid, kl_divergence
    #     "metric_name": "Mean Squared Error",  # Mean Squared Error, KL Divergence
    #     "cutoff_steps": 1e-2,
    # },
}
# hm(s): heatmap as collection (single)
# sce: scatter plots of fcc and expressibility over mse
# rel: coefficient-parameter relation
# var: coefficient variance
# dist: distribution plot (for classical training)
enabled_plots = ["var"]  # "bp", "sc", "sce", "hm", "hms"

all_metrics = ["mse_valid", "huber_loss_valid", "kl_divergence_valid"]

for scenario, setting in scenarios.items():
    print(f"{'-' * 100}")
    print(f"\nScenario: {scenario}\n")
    print(f"{'-' * 100}")

    metric = setting["metric"]
    metric_name = setting["metric_name"]
    cutoff_steps = setting["cutoff_steps"]

    # get run_ids
    coefficient_run_ids = get_run_ids(setting["coefficient_id"])
    expr_run_ids = get_run_ids(setting["expr_id"])
    training_run_ids = get_run_ids(setting["training_experiment_id"])
    if "classical_training_experiment_id" in setting:
        classical_training_run_ids = get_run_ids(
            setting["classical_training_experiment_id"]
        )
        cache_id = setting

        df = cache_df(cache_id)
    else:
        classical_training_run_ids = None

        cache_id = setting

        df = cache_df(cache_id)

    if not cache or df is None:

        # get dataframes
        coefficients_df = get_coefficient_df(coefficient_run_ids)
        expr_df = get_expressibility_df(expr_run_ids)
        training_df = get_training_df(
            training_run_ids,
            cutoff_steps=cutoff_steps,
            metrics=all_metrics,
            run_id_tag="training_run_id",
        )
        classical_training_df = get_training_df(
            classical_training_run_ids,
            cutoff_steps=cutoff_steps,
            metrics=all_metrics,
            run_id_tag="classical_training_run_id",
        )

        # combine dataframes
        if training_df is None and expr_df is None:
            combined_df = coefficients_df
        else:
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
                "coeff_var_real",
                "coeff_var_imag",
                "coeff_var_abs",
                "coeff_mean_real",
                "coeff_mean_imag",
                "coeff_mean_abs",
                "expressibility",
                "steps",
                "steps_var",
                *[f"{metric}_min" for metric in all_metrics],
                *[f"{metric}_var" for metric in all_metrics],
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
                    df.loc[idx, "seed"] = seed
                    df.loc[idx, "coeff_run_id"] = current_dataset.coeff_run_id.unique()[
                        0
                    ]
                    df.at[idx, "coeff_var_real"] = (
                        current_dataset.coeff_var_real.mean().tolist()
                    )
                    df.at[idx, "coeff_var_imag"] = (
                        current_dataset.coeff_var_imag.mean().tolist()
                    )
                    df.at[idx, "coeff_var_abs"] = (
                        current_dataset.coeff_var_abs.mean().tolist()
                    )
                    df.at[idx, "coeff_mean_real"] = (
                        current_dataset.coeff_mean_real.mean().tolist()
                    )
                    df.at[idx, "coeff_mean_imag"] = (
                        current_dataset.coeff_mean_imag.mean().tolist()
                    )
                    df.at[idx, "coeff_mean_abs"] = (
                        current_dataset.coeff_mean_abs.mean().tolist()
                    )

                    if training_df is not None:
                        df.loc[idx, "steps"] = current_dataset.steps.mean()
                        df.loc[idx, "steps_var"] = current_dataset.steps.var()
                        df.loc[idx, f"{metric}_min"] = current_dataset[
                            f"{metric}_min"
                        ].mean()
                        df.loc[idx, f"{metric}_min_var"] = current_dataset[
                            f"{metric}_min"
                        ].var()
                        df.loc[idx, "training_run_id"] = (
                            f"{current_dataset.training_run_id.to_list()}"
                        )
                        if classical_training_df is not None:
                            df.loc[idx, "classical_training_run_id"] = (
                                f"{classical_training_df[classical_training_df.seed==seed].training_run_id.to_list()}"
                            )
                    if expr_df is not None:
                        df.loc[idx, "expressibility"] = (
                            current_dataset.expressibility.mean()
                        )

                        df.loc[idx, "expr_run_id"] = (
                            current_dataset.expr_run_id.unique()[0]
                        )

                    idx += 1

        print(f"Caching dataframe: {df.describe()}")

        if classical_training_df is not None:
            cache_df(
                run_ids=cache_id,
                df=df,
            )
        else:
            cache_df(run_ids=cache_id, df=df)
    else:
        print(f"Using cached dataframe: {df.describe()}")
        # get unique values
        qubits = df.qubits.unique()
        ansaetze = df.ansatz.unique()
        ansatz_ids = df.ansatz_id.unique()

    for q in qubits:
        if "bp" in enabled_plots:
            fig = visualize_boxplot(df[df.qubits == q], metric)
            fig.update_layout(title=f"{scenario}, {q} Qubits, Metric: {metric}")
            save_fig(
                fig,
                f"{scenario}_bp_q{q}",
                cache_id,
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
                f"{scenario}_sc_q{q}_{'w' if weighted else 'uw'}",
                cache_id,
                scenario,
            )

        if "sce" in enabled_plots:
            # scatter plot
            fig = visualize_expr_scatter(
                df[df.qubits == q],
                ansatz_ids,
                f"{metric}_min",
                metric_name,
                weighted=weighted,
                legendonly=False,
            )
            fig.update_layout(title=f"{scenario}, {q} Qubits, Metric: {metric})")
            fig.update_layout(title=f"")
            save_fig(
                fig,
                f"{scenario}_sce_q{q}_{'w' if weighted else 'uw'}",
                cache_id,
                scenario,
                showlegend=False,
            )

            # legendonly
            fig = visualize_expr_scatter(
                df[df.qubits == q],
                ansatz_ids,
                f"{metric}_min",
                metric_name,
                weighted=weighted,
                legendonly=True,
            )
            fig.update_layout(title=f"{scenario}, {q} Qubits, Metric: {metric})")
            fig.update_layout(title=f"")
            save_fig(
                fig,
                f"{scenario}_sce_q{q}_{'w' if weighted else 'uw'}_legend",
                cache_id,
                scenario,
                showlegend=True,
                font_size=20,
            )

        if scenario == "1dfs_rc":
            fig = visualize_single_heatmap(
                df[df.qubits == q],
                1000,
                f"random_coefficients_correlated",
            )
            fig.update_layout(title=f"")
            save_fig(
                fig,
                f"{scenario}_hm_q{q}_{'w' if weighted else 'uw'}",
                cache_id,
                scenario,
            )
        elif "1dfs" in scenario and "hm" in enabled_plots:
            fig = visualize_heatmap(df[df.qubits == q], 1000, weighted)
            fig.update_layout(title=f"")
            save_fig(
                fig,
                f"{scenario}_hm_q{q}_{'w' if weighted else 'uw'}",
                cache_id,
                scenario,
            )

        if "1dfs" in scenario and "rel" in enabled_plots:
            fig = visualize_coeff_param_relation(df[df.qubits == q], 1000)
            fig.update_layout(title=f"")
            save_fig(
                fig,
                f"{scenario}_rel_q{q}_{'w' if weighted else 'uw'}",
                cache_id,
                scenario,
                # font_size=20,
            )

        if "1dfs" in scenario and "var" in enabled_plots:
            fig = visualize_coeff_variance(
                df[df.qubits == q], weighted=False, legendonly=False
            )
            fig.update_layout(title=f"")
            save_fig(
                fig,
                f"{scenario}_var_q{q}_{'w' if weighted else 'uw'}",
                cache_id,
                scenario,
                showlegend=False,
            )

            # legendonly
            fig = visualize_coeff_variance(
                df[df.qubits == q], weighted=False, legendonly=True
            )
            fig.update_layout(title=f"")
            save_fig(
                fig,
                f"{scenario}_sce_q{q}_{'w' if weighted else 'uw'}_legend",
                cache_id,
                scenario,
                showlegend=True,
                font_size=20,
            )

        if "2dfs" in scenario and "hms" in enabled_plots:
            fig = visualize_single_heatmap(
                df[(df.qubits == q) & (df.ansatz == "Hardware_Efficient")],
                1000,
                f"coefficients_correlated{'_weighted' if weighted else ''}",
            )
            fig.update_layout(title=f"")
            save_fig(
                fig,
                f"{scenario}_hms_q{q}_{'w' if weighted else 'uw'}",
                cache_id,
                scenario,
            )

        if "2dhepc" in scenario and "dist" in enabled_plots:
            fig = visualize_distribution(
                df[df.qubits == q], identifier="fig_distribution_train"
            )
            fig.update_layout(title=f"")
            save_fig(
                fig,
                f"{scenario}_dist_train_q{q}",
                cache_id,
                scenario,
            )

            fig = visualize_distribution(
                df[df.qubits == q], identifier="fig_distribution_valid"
            )
            fig.update_layout(title=f"")
            save_fig(
                fig,
                f"{scenario}_dist_valid_q{q}",
                cache_id,
                scenario,
            )
