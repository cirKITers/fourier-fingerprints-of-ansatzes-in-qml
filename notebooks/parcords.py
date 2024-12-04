import plotly
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio
from runs.training_runs import run_ids as training_run_ids
from runs.training_runs import experiment_id
from runs.coefficient_runs import run_ids as coefficient_run_ids
from helper import save_fig, get_training_df, get_coefficient_df, assign_ansatz_id

pio.kaleido.scope.mathjax = None

training_df = get_training_df(training_run_ids)

coefficients_df = get_coefficient_df(coefficient_run_ids)

combined_df = pd.merge(training_df, coefficients_df, on=["ansatz", "qubits"])
combined_df.sort_values(by="qubits", inplace=True)
combined_df = assign_ansatz_id(combined_df)

qubits = combined_df.qubits.unique()
ansaetze = combined_df.ansatz.unique()
ansatz_ids = combined_df.ansatz_id.unique()

for q in qubits:

    fig = go.Figure(
        data=[
            go.Parcoords(
                line=dict(
                    color=combined_df[combined_df.qubits == q].ansatz_id,
                    colorscale=plotly.colors.qualitative.T10_r,
                    # showscale=True,
                    cmin=0,
                    cmax=len(ansaetze),
                ),
                dimensions=list(
                    [
                        dict(
                            label="Ansatz",
                            values=combined_df[combined_df.qubits == q].ansatz_id,
                            tickvals=ansatz_ids,
                            ticktext=ansaetze,
                        ),
                        # dict(label="# Qubits", values=df["qubits"]),
                        # dict(
                        #     label="coefficients_correlation_variance",
                        #     values=df["coefficients_correlation_variance"],
                        # ),
                        # dict(
                        #     label="# Parameters",
                        #     values=df["n_params"],
                        # ),
                        dict(
                            label="Expressibility",
                            values=combined_df[combined_df.qubits == q].expressibility,
                        ),
                        dict(
                            label="Coeff. Correlation Mean",
                            values=combined_df[
                                combined_df.qubits == q
                            ].coefficients_correlation_mean,
                        ),
                        # dict(label="mse (log)", values=df[df.qubits == q].mse),
                        dict(
                            label="Steps",
                            values=combined_df[combined_df.qubits == q].steps,
                        ),
                    ]
                ),
            )
        ]
    )

    fig.update_layout(
        title=f"Correlation of Coefficients, Expressibility and Loss ({q} Qubits)",
        # template="plotly_white",
        margin=dict(l=120),
    )

    save_fig(
        fig, f"parcords_q{q}", coefficient_run_ids + training_run_ids, experiment_id
    )
