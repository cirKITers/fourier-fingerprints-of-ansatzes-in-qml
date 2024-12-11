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

main_colors_it, sec_colors_it = get_color_iterator()


fig = go.Figure()
for i, metric in enumerate(["expressibility"]):
    main_color_sel = next(main_colors_it)
    sec_color_sel = rgb_to_rgba(next(sec_colors_it), 0.2)

    metric_values = coefficient_df.groupby("layer_multiplier")[metric].agg(
        ["mean", "min", "max"]
    )

    fig.add_trace(
        go.Scatter(
            x=metric_values.index,
            y=metric_values["mean"],
            name=metric,
            mode="lines",
            line=dict(color=main_color_sel),
            marker=dict(color=main_color_sel),
            yaxis=f"y{i+1}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=metric_values.index,
            y=metric_values["max"],
            name=f"upper-{metric}",
            visible=True,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            yaxis=f"y{i+1}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=metric_values.index,
            y=metric_values["min"],
            name=f"lower-{metric}",
            visible=True,
            mode="lines",
            fill="tonexty",
            fillcolor=sec_color_sel,
            marker=dict(color=main_color_sel),
            line=dict(width=0),
            showlegend=False,
            yaxis=f"y{i+1}",
        )
    )

fig.update_layout(
    title=f"Coefficient Correlation and Expressibility over Layers",
    template="plotly_white",
    yaxis=dict(title="Expressibility"),
    yaxis2=dict(title="Coefficient Correlation Mean", overlaying="y", side="right"),
    xaxis=dict(title="Layers"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

save_fig(fig, "coefficient_correlation_layers", run_ids, experiment_id)
