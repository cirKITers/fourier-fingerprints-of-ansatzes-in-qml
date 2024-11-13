import mlflow
import plotly
import plotly.graph_objects as go
import re
import json
import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.express as px
from training_runs import run_ids as training_run_ids
from coefficient_runs import run_ids as coefficient_run_ids
from helper import generate_hash
import hashlib

pio.kaleido.scope.mathjax = None


def read_from_csv(path):
    with open(path) as f:
        return pd.read_csv(f)


global_training_df = pd.DataFrame()
all_ansaetze = []
all_qubits = []
seeds = []
for it, run_id in enumerate(training_run_ids):
    client = mlflow.tracking.MlflowClient()
    all_ansaetze.append(client.get_run(run_id).data.params["model.circuit_type"])
    all_qubits.append(int(client.get_run(run_id).data.params["model.n_qubits"]))
    seeds.append(int(client.get_run(run_id).data.params["seed"]))
ansaetze = list(set(all_ansaetze))
qubits = list(set(all_qubits))
n_ansaetze = len(set(all_ansaetze))

global_training_df["ansatz"] = all_ansaetze
global_training_df["qubits"] = all_qubits
global_training_df["run_id"] = training_run_ids

df = pd.DataFrame(
    columns=[
        "ansatz",
        "n_qubits",
        "expressibility",
        "mse",
        "coefficients_correlation_mean",
        # "coefficients_correlation_variance",
    ]
)

mse_qubit_ansatz = {}
for q in qubits:
    fig = go.Figure()
    main_colors_it = iter(plotly.colors.qualitative.Dark2)
    sec_colors_it = iter(plotly.colors.qualitative.Pastel2)

    mse_qubit_ansatz[q] = {}
    for ansatz in ansaetze:
        max_steps = 0
        mse_seeds = []
        # iterate global_df where qubits and ansatz match q and ansatz
        for index, row in global_training_df[
            (global_training_df.qubits == q) & (global_training_df.ansatz == ansatz)
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

        shortest = np.array(min(mse_seeds, key=lambda x: len(x)))
        mse_low = np.nanmin(np_mse, axis=0)
        mse_var = np.nanstd(np_mse, axis=0)
        mse_high = np.nanmax(np_mse, axis=0)
        mse_mean = np.nanmean(np_mse, axis=0)

        mse_qubit_ansatz[q][ansatz] = np.min(mse_mean)


for it, run_id in enumerate(coefficient_run_ids):
    client = mlflow.tracking.MlflowClient()
    if client.get_run(run_id).info.status != "FINISHED":
        print(f"Run {run_id} not finished")
        continue
    # if int(client.get_run(run_id).data.params["n_qubits"]) != 7:
    #     continue

    # csv_path = client.download_artifacts(run_id, f"params.csv", "./")
    # try:
    #     params = read_from_csv(csv_path)
    #     df.loc[it, "n_params"] = len(params[params["step"] == 0])
    # except:
    #     df.loc[it, "n_params"] = 0

    df.loc[it, "ansatz"] = ansaetze.index(
        client.get_run(run_id).data.params["model.circuit_type"]
    )

    df.loc[it, "n_qubits"] = int(client.get_run(run_id).data.params["model.n_qubits"])

    # df.loc[it, "expressibility"] = client.get_run(run_id).data.metrics["expressibility"]

    df.loc[it, "mse"] = np.log(
        mse_qubit_ansatz[df.loc[it, "n_qubits"]][ansaetze[df.loc[it, "ansatz"]]]
    )

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
for q in qubits:

    fig = go.Figure(
        data=[
            go.Parcoords(
                line=dict(
                    color=df[df.n_qubits == q].ansatz,
                    colorscale=plotly.colors.qualitative.T10_r,
                    # showscale=True,
                    cmin=0,
                    cmax=len(ansaetze),
                ),
                dimensions=list(
                    [
                        dict(
                            label="Ansatz",
                            values=df[df.n_qubits == q].ansatz,
                            tickvals=list(set(df[df.n_qubits == q].ansatz)),
                            ticktext=ansaetze,
                        ),
                        # dict(label="# Qubits", values=df["n_qubits"]),
                        # dict(
                        #     label="coefficients_correlation_variance",
                        #     values=df["coefficients_correlation_variance"],
                        # ),
                        # dict(
                        #     label="# Parameters",
                        #     values=df["n_params"],
                        # ),
                        # dict(
                        #     label="Expressibility",
                        #     values=df[df.n_qubits == q].expressibility,
                        # ),
                        dict(
                            label="Coeff. Correlation Mean",
                            values=df[df.n_qubits == q].coefficients_correlation_mean,
                        ),
                        dict(label="mse (log)", values=df[df.n_qubits == q].mse),
                    ]
                ),
            )
        ]
    )

    fig.update_layout(
        title=f"Correlation of Coefficients & MSE ({q} Qubits)",
        # template="plotly_white",
        margin=dict(l=120),
    )

    hs = generate_hash(coefficient_run_ids)

    # fig.show()
    fig.write_image(f"results/parcords_q{q}_{hs}.png")
