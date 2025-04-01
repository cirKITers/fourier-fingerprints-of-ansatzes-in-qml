from plotly.subplots import make_subplots
import plotly.io as pio
from runs.coefficient_runs import run_ids, experiment_id
from helper import (
    get_coefficient_df,
    assign_ansatz_id,
    save_fig,
    get_plotly_artifact,
)

pio.kaleido.scope.mathjax = None

selected_seed = 1000

coefficient_df = get_coefficient_df(run_ids)
coefficient_df.sort_values(by="ansatz", inplace=True)
coefficient_df = assign_ansatz_id(coefficient_df)

qubits = coefficient_df.qubits.unique()
ansaetze = coefficient_df.ansatz.unique()

# ----------------------------------

for q in qubits:
    fig = make_subplots(rows=1, cols=len(ansaetze), subplot_titles=ansaetze)

    for it, ansatz in enumerate(ansaetze):

        sub_fig_trace = get_plotly_artifact(
            coefficient_df[
                (coefficient_df.qubits == q)
                & (coefficient_df.ansatz == ansatz)
                & (coefficient_df.seed == selected_seed)
            ].run_id.item(),
            "spectrum",
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
        title_text=(f"Spectrum for different Ansaetze ({q} Qubits)"),
        template="plotly_white",
        height=400,
        width=300 * it,
    )

    save_fig(
        fig,
        (f"coefficients_correlated_q{q}"),
        run_ids,
        experiment_id,
    )
