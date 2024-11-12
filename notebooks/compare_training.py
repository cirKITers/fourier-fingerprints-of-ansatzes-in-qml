import mlflow
import plotly
import plotly.graph_objects as go
import re
import json
import pandas as pd
import numpy as np
import plotly.io as pio
from training_runs import run_ids, experiment_id
from helper import generate_hash

pio.kaleido.scope.mathjax = None


def rgb_to_rgba(rgb_value: str, alpha: float):
    """
    Adds the alpha channel to an RGB Value and returns it as an RGBA Value
    :param rgb_value: Input RGB Value
    :param alpha: Alpha Value to add  in range [0,1]
    :return: RGBA Value
    """
    return f"rgba{rgb_value[3:-1]}, {alpha})"


def read_from_html(path):
    with open(path) as f:
        html = f.read()
    call_arg_str = re.findall(r"Plotly\.newPlot\((.*)\)", html[-(2**16) :])[0]
    call_args = json.loads(f"[{call_arg_str}]")
    plotly_json = {"data": call_args[1], "layout": call_args[2]}
    return plotly.io.from_json(json.dumps(plotly_json))


global_df = pd.DataFrame()
all_ansaetze = []
all_qubits = []
seeds = []
for it, run_id in enumerate(run_ids):
    client = mlflow.tracking.MlflowClient()
    all_ansaetze.append(client.get_run(run_id).data.params["model.circuit_type"])
    all_qubits.append(int(client.get_run(run_id).data.params["model.n_qubits"]))
    seeds.append(int(client.get_run(run_id).data.params["seed"]))
ansaetze = list(set(all_ansaetze))
qubits = list(set(all_qubits))
n_ansaetze = len(set(all_ansaetze))

global_df["ansatz"] = all_ansaetze
global_df["qubits"] = all_qubits
global_df["run_id"] = run_ids

# ----------------------------------


for q in qubits:
    fig = go.Figure()
    main_colors_it = iter(plotly.colors.qualitative.Dark2)
    sec_colors_it = iter(plotly.colors.qualitative.Pastel2)

    for ansatz in ansaetze:
        max_steps = 0
        mse_seeds = []
        # iterate global_df where qubits and ansatz match q and ansatz
        for index, row in global_df[
            (global_df.qubits == q) & (global_df.ansatz == ansatz)
        ].iterrows():

            client = mlflow.tracking.MlflowClient()

            mse_hist = client.get_metric_history(row.run_id, "mse")
            max_steps = max(max_steps, len(mse_hist))
            mse = [entity.value for entity in mse_hist]
            mse_seeds.append(mse)

        np_mse = np.zeros([len(mse_seeds), len(max(mse_seeds, key=lambda x: len(x)))])
        for i, j in enumerate(mse_seeds):
            np_mse[i][0 : len(j)] = j
            np_mse[i][len(j) :] = np.nan

        mse_low = np.nanmin(np_mse, axis=0)
        mse_high = np.nanmax(np_mse, axis=0)
        mse_mean = np.nanmean(np_mse, axis=0)

        main_color_sel = next(main_colors_it)
        sec_color_sel = rgb_to_rgba(next(sec_colors_it), 0.2)

        # now add scatter plot for this ansatz and qubit with error bands as mse_low and mse_high
        fig.add_trace(
            go.Scatter(
                x=list(range(mse_mean.size)),
                y=mse_mean,
                name=f"{ansatz}",
                visible=True,
                mode="lines",
                line=dict(color=main_color_sel),
                marker=dict(color=main_color_sel),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(mse_high.size)),
                y=mse_high,
                name=f"upper-{ansatz}",
                visible=True,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(mse_low.size)),
                y=mse_low,
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
        title=f"Loss for Different Ansaetze ({q} Qubits)",
        template="plotly_white",
        yaxis=dict(title="MSE"),
        xaxis=dict(title="Epochs"),
        # sliders=sliders,
    )

    fig.show()
    hs = generate_hash(run_ids)
    fig.write_image(f"mse_q{q}_{hs}.pdf")
