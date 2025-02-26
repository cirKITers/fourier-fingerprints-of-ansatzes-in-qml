import plotly
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
from runs.training_runs import run_ids, experiment_id
from helper import save_fig, get_training_df, rgb_to_rgba, get_color_iterator

pio.kaleido.scope.mathjax = None

max_steps = 1000

cutoff = 1e-4

training_df = get_training_df(run_ids, cutoff_mse=cutoff)
training_df.sort_values(by="qubits", inplace=True)

qubits = training_df.qubits.unique()
ansaetze = training_df.ansatz.unique()


for qubit in qubits:
    fig = go.Figure()
    main_colors_it, sec_colors_it = get_color_iterator()

    for ansatz in ansaetze:
        main_color_sel = next(main_colors_it)
        sec_color_sel = rgb_to_rgba(next(sec_colors_it), 0.2)

        mse_values = np.stack(
            training_df[
                (training_df.qubits == qubit) & (training_df.ansatz == ansatz)
            ].mse.values
        )
        if max_steps is not None:
            steps = list(range(max_steps))
            mse_values = mse_values[:, :max_steps]
        else:
            steps = list(range(mse_values.shape[1]))

        # now add scatter plot for this ansatz and qubit with error bands as mse_low and mse_high
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=np.nanmean(mse_values, axis=0),
                name=f"{ansatz}",
                visible=True,
                mode="lines",
                line=dict(color=main_color_sel),
                marker=dict(color=main_color_sel),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=np.nanmax(mse_values, axis=0),
                name=f"upper-{ansatz}",
                visible=True,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=np.nanmin(mse_values, axis=0),
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

    fig.update_layout(
        title=f"Loss for Different Ansaetze ({qubit} Qubits)",
        template="plotly_white",
        yaxis=dict(title="MSE", type="log"),
        xaxis=dict(title="Epochs"),
        # sliders=sliders,
        hovermode="x",
        showlegend=True,
    )

    save_fig(fig, f"mse_q{qubit}_c{cutoff}", run_ids, experiment_id)
