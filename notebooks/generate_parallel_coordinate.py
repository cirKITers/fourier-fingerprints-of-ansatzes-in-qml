import mlflow
import plotly
import plotly.graph_objects as go
import re
import json
import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.express as px
from notebooks.runs import run_ids, experiment_id
import hashlib

pio.kaleido.scope.mathjax = None


def read_from_csv(path):
    with open(path) as f:
        return pd.read_csv(f)


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
        "n_params",
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
    if int(client.get_run(run_id).data.params["n_qubits"]) != 7:
        continue

    csv_path = client.download_artifacts(run_id, f"params.csv", "./")
    try:
        params = read_from_csv(csv_path)
        df.loc[it, "n_params"] = len(params[params["step"] == 0])
    except:
        df.loc[it, "n_params"] = 0

    df.loc[it, "ansatz"] = ansaetze.index(
        client.get_run(run_id).data.params["circuit_type"]
    )

    df.loc[it, "n_qubits"] = int(client.get_run(run_id).data.params["n_qubits"])
    df.loc[it, "mse"] = np.log(client.get_run(run_id).data.metrics["mse"])
    # df.loc[it, "mse"] = client.get_run(run_id).data.metrics["mse"]
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
                colorscale=plotly.colors.qualitative.T10_r,
                # showscale=True,
                cmin=0,
                cmax=len(ansaetze),
            ),
            dimensions=list(
                [
                    dict(
                        label="Ansatz",
                        values=df["ansatz"],
                        tickvals=list(set(df["ansatz"])),
                        ticktext=ansaetze,
                    ),
                    # dict(label="# Qubits", values=df["n_qubits"]),
                    # dict(
                    #     label="coefficients_correlation_variance",
                    #     values=df["coefficients_correlation_variance"],
                    # ),
                    dict(
                        label="# Parameters",
                        values=df["n_params"],
                    ),
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
    # template="plotly_white",
    margin=dict(l=120),
)

hs = hashlib.md5(repr(run_ids).encode("utf-8")).hexdigest()
filename = f"parcoords_{hs}.pdf"

print(f"Output to {filename}")
fig.write_image(filename)
