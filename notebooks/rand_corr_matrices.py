from plotly.subplots import make_subplots
from plotly import graph_objects as go
import plotly.io as pio
from runs.coefficient_runs import run_ids, experiment_id
from helper import (
    get_coefficient_df,
    assign_ansatz_id,
    save_fig,
    get_correlation_matrix,
)

pio.kaleido.scope.mathjax = None

selected_seed = 1004
selected_ansatz = "Strongly_Entangling"

coefficient_df = get_coefficient_df(run_ids)
coefficient_df.sort_values(by="qubits", inplace=True)
coefficient_df = assign_ansatz_id(coefficient_df)

qubits = coefficient_df.qubits.unique()

# ----------------------------------

fig = make_subplots(
    rows=1, cols=len(qubits), subplot_titles=[f"{q} Qubits" for q in qubits]
)
for it, q in enumerate(qubits):

    sub_fig_trace = get_correlation_matrix(
        coefficient_df[
            (coefficient_df.qubits == q)
            & (coefficient_df.ansatz == selected_ansatz)
            & (coefficient_df.seed == selected_seed)
        ].run_id.item(),
        identifier="random_coefficients_correlated",
    )

    fig.add_trace(sub_fig_trace, row=1, col=it + 1)
    fig.update_xaxes(dict(title="Coefficients"), row=1, col=it + 1)
    fig.update_yaxes(
        dict(
            title="Coefficients" if it == 0 else "",
            autorange="reversed",
            scaleanchor="x",
        ),
        row=1,
        col=it + 1,
    )

fig.update_layout(
    title_text=f"Correlation of Random Coefficients for different Qubits",
    template="plotly_white",
    height=400,
    width=300 * it,
    coloraxis=dict(
        colorscale="Sunset",
    ),
)

save_fig(
    fig, f"random_coefficients_correlated_{selected_ansatz}", run_ids, experiment_id
)
