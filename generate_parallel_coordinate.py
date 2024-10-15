import mlflow
import plotly
import plotly.graph_objects as go
import re
import json
import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.express as px
from runs import run_ids, experiment_id
import hashlib

pio.kaleido.scope.mathjax = None


def read_from_html(path):
    with open(path) as f:
        html = f.read()
    call_arg_str = re.findall(r"Plotly\.newPlot\((.*)\)", html[-(2**16) :])[0]
    call_args = json.loads(f"[{call_arg_str}]")
    plotly_json = {"data": call_args[1], "layout": call_args[2]}
    return plotly.io.from_json(json.dumps(plotly_json))


ansaetze = []
for it, run_id in enumerate(run_ids):
    client = mlflow.tracking.MlflowClient()
    ansatz = client.get_run(run_id).data.params["circuit_type"]
    if ansatz not in ansaetze:
        ansaetze.append(ansatz)


df = pd.DataFrame(
    columns=[
        "ansatz",
        "n_qubits",
        "mse",
        "coefficients_correlation_mean",
        # "coefficients_correlation_variance",
    ]
)

for it, run_id in enumerate(run_ids):
    client = mlflow.tracking.MlflowClient()
    if client.get_run(run_id).info.status != "FINISHED":
        print(f"Run {run_id} not finished")
        continue
    if int(client.get_run(run_id).data.params["n_qubits"]) != 6:
        continue

    df.loc[it, "ansatz"] = ansaetze.index(
        client.get_run(run_id).data.params["circuit_type"]
    )

    df.loc[it, "n_qubits"] = int(client.get_run(run_id).data.params["n_qubits"])
    # df.loc[it, "mse"] = np.log(client.get_run(run_id).data.metrics["mse"])
    df.loc[it, "mse"] = np.log(client.get_run(run_id).data.metrics["mse"])
    df.loc[it, "coefficients_correlation_mean"] = client.get_run(run_id).data.metrics[
        "coefficients_correlation_mean"
    ]

    # df.loc[it, "coefficients_correlation_variance"] = client.get_run(
    #     run_id
    # ).data.metrics["coefficients_correlation_variance"]
    # df[ansaetze[it]] = [
    #     client.get_run(run_id).data.metrics["mse"],
    #     client.get_run(run_id).data.metrics["coefficients_correlation_mean"],
    #     client.get_run(run_id).data.metrics["coefficients_correlation_variance"],
    # ]
# df.sort_values(by="mse", inplace=True)
fig = go.Figure(
    data=[
        go.Parcoords(
            line=dict(
                color=df["ansatz"],
                colorscale=px.colors.qualitative.T10,
            ),
            dimensions=list(
                [
                    dict(
                        label="Ansatz",
                        values=df["ansatz"],
                        tickvals=list(set(df["ansatz"])),
                        ticktext=ansaetze,
                    ),
                    dict(label="# Qubits", values=df["n_qubits"]),
                    # dict(
                    #     label="coefficients_correlation_variance",
                    #     values=df["coefficients_correlation_variance"],
                    # ),
                    dict(
                        label="Coeff. Correlation Mean",
                        values=df["coefficients_correlation_mean"],
                    ),
                    dict(label="mse (log)", values=df["mse"]),
                ]
            ),
        )
    ]
)

fig.update_layout(
    title="Parallel Coordinats: Correlation of Coefficients & MSE",
    template="plotly_white",
    margin=dict(l=120),
)

hs = hashlib.md5(repr(run_ids).encode("utf-8")).hexdigest()
filename = f"parcoords_{hs}.pdf"

print(f"Output to {filename}")
fig.write_image(filename)
