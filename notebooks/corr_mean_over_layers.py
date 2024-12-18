import plotly.graph_objects as go
import plotly.io as pio
from runs.coeffexpr_runs import run_ids, experiment_id
from helper import (
    save_fig,
    get_coefficient_df,
    rgb_to_rgba,
    get_color_iterator,
    assign_ansatz_id,
)

pio.kaleido.scope.mathjax = None

coefficient_df = get_coefficient_df(run_ids)
coefficient_df.sort_values(by="qubits", inplace=True)
coefficient_df = assign_ansatz_id(coefficient_df)

ansaetze = coefficient_df.ansatz.unique()


for metric in [
    "coefficients_correlation_mean",
    "coefficients_correlation_max",
    "kl_divergence",
]:
    fig = go.Figure()
    main_colors_it, sec_colors_it = get_color_iterator()
    for ansatz in ansaetze:
        main_color_sel = next(main_colors_it)
        sec_color_sel = rgb_to_rgba(next(sec_colors_it), 0.2)

        metric_values = (
            coefficient_df[coefficient_df.ansatz == ansatz]
            .groupby("layer_multiplier")[metric]
            .agg(["mean", "min", "max"])
        )

        fig.add_trace(
            go.Scatter(
                x=metric_values.index,
                y=metric_values["mean"],
                name=ansatz,
                mode="lines",
                line=dict(color=main_color_sel),
                marker=dict(color=main_color_sel),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=metric_values.index,
                y=metric_values["max"],
                name=f"upper-{ansatz}",
                visible=True,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=metric_values.index,
                y=metric_values["min"],
                name=f"lower-{ansatz}",
                visible=True,
                mode="lines",
                fill="tonexty",
                fillcolor=sec_color_sel,
                marker=dict(color=main_color_sel),
                line=dict(width=0),
                showlegend=False,
            )
        )

    title = metric.replace("_", " ").title()
    fig.update_layout(
        title=f"{title} over Additional Layers",
        template="plotly_white",
        yaxis=dict(title=f"{title} (log)", type="log"),
        xaxis=dict(title="Layer Multiplier"),
    )

    save_fig(fig, f"{metric}_layer_multiplier", run_ids, experiment_id)
