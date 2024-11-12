import mlflow
import plotly
import plotly.graph_objects as go
import re
import json
import pandas as pd
import numpy as np
import plotly.io as pio
from notebooks.training_runs import run_ids, experiment_id

pio.kaleido.scope.mathjax = None


def read_from_html(path):
    with open(path) as f:
        html = f.read()
    call_arg_str = re.findall(r"Plotly\.newPlot\((.*)\)", html[-(2**16) :])[0]
    call_args = json.loads(f"[{call_arg_str}]")
    plotly_json = {"data": call_args[1], "layout": call_args[2]}
    return plotly.io.from_json(json.dumps(plotly_json))


global_df = pd.DataFrame()
all_ansaetze = []
qubits = []
for it, run_id in enumerate(run_ids):
    client = mlflow.tracking.MlflowClient()
    all_ansaetze.append(client.get_run(run_id).data.params["circuit_type"])
    qubits.append(int(client.get_run(run_id).data.params["n_qubits"]))
ansaetze = list(set(all_ansaetze))
n_ansaetze = len(set(all_ansaetze))

global_df["ansatz"] = all_ansaetze
global_df["qubits"] = qubits
global_df["run_id"] = run_ids

# ----------------------------------

fig = go.Figure()

for q in qubits:
    loss_precision = 1e3
    df = pd.DataFrame()

    for index, row in global_df[global_df.qubits == q].iterrows():

        client = mlflow.tracking.MlflowClient()

        mse_hist = client.get_metric_history(row.run_id, "mse")
        df[row.ansatz] = [
            np.trunc(entity.value * loss_precision) / loss_precision
            for entity in mse_hist
        ]

    fig.add_traces(
        [go.Scatter(x=df.index, y=df[ansatz], name=f"{ansatz}-q{q}", visible=False) for ansatz in ansaetze]
    )

fig.data[:-n_ansaetze].visible = True

# Create and add slider
steps = []
for q in qubits:
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(q)}],  # layout attribute
    )
    step["args"][0]["visible"][q*n_ansaetze:(q+1)*n_ansaetze] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=qubits,
    currentvalue={"prefix": "Qubits: "},
    # pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    title=f"Loss for Different Ansaetze ({q} Qubits)",
    template="plotly_white",
    yaxis=dict(
        title="MSE", type="log", range=[np.log(1 / loss_precision), np.log(1)]
    ),
    xaxis=dict(title="Epochs", type="log"),
    sliders=sliders
)

fig.write_image(f"mse_q{q}.pdf")
