import mlflow
import plotly
import plotly.graph_objects as go
import re
import json
import pandas as pd
import numpy as np
import plotly.io as pio
from runs import run_ids, experiment_id

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
qubits=[]
for it, run_id in enumerate(run_ids):
    client = mlflow.tracking.MlflowClient()
    all_ansaetze.append(client.get_run(run_id).data.params["circuit_type"])
    qubits.append(int(client.get_run(run_id).data.params["n_qubits"]))
ansaetze = list(set(all_ansaetze))
n_ansaetze = len(set(all_ansaetze))

global_df['ansatz'] = all_ansaetze
global_df['qubits'] = qubits
global_df['run_id'] = run_ids

# ----------------------------------

for q in qubits:
    loss_precision = 1e3
    df = pd.DataFrame()

    for index, row in global_df[global_df.qubits==q].iterrows():

        client = mlflow.tracking.MlflowClient()

        mse_hist = client.get_metric_history(row.run_id, "mse")
        df[row.ansatz] = [
            np.trunc(entity.value * loss_precision) / loss_precision for entity in mse_hist
        ]


    fig = go.Figure(
        data=[go.Scatter(x=df.index, y=df[ansatz], name=ansatz) for ansatz in ansaetze]
    )

    fig.update_layout(
        title=f"Loss for Different Ansaetze ({q} Qubits)",
        template="plotly_white",
        yaxis=dict(title="MSE", type="log", range=[np.log(1 / loss_precision), np.log(1)]),
        xaxis=dict(title="Epochs", type="log"),
    )

    fig.write_image(f"mse_q{q}.pdf")
