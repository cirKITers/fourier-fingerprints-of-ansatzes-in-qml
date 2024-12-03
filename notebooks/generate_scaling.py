import mlflow
import plotly
import plotly.graph_objects as go
import re
import json
import pandas as pd
import numpy as np
import plotly.io as pio
import hashlib
from coefficient_runs import run_ids
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
qubits = []
for it, run_id in enumerate(run_ids):
    client = mlflow.tracking.MlflowClient()
    all_ansaetze.append(client.get_run(run_id).data.params["model.circuit_type"])
    qubits.append(int(client.get_run(run_id).data.params["model.n_qubits"]))
ansaetze = list(set(all_ansaetze))
n_qubits = list(set(qubits))

# ----------------------------------

df = pd.DataFrame(
    columns=[
        "ansatz",
        "n_qubits",
        "coefficients_correlation_mean",
        # "coefficients_correlation_variance",
    ]
)

for it, run_id in enumerate(run_ids):

    client = mlflow.tracking.MlflowClient()
    if client.get_run(run_id).info.status != "FINISHED":
        print(f"Run {run_id} not finished")
        continue

    df.loc[it, "ansatz"] = ansaetze.index(
        client.get_run(run_id).data.params["model.circuit_type"]
    )
    df.loc[it, "n_qubits"] = int(client.get_run(run_id).data.params["model.n_qubits"])
    # df.loc[it, "mse"] = np.log(client.get_run(run_id).data.metrics["mse"])
    df.loc[it, "coefficients_correlation_mean"] = client.get_run(run_id).data.metrics[
        "coefficients_correlation_mean"
    ]

df.sort_values(by="n_qubits", inplace=True)

main_colors_it = iter(plotly.colors.qualitative.Dark2)
sec_colors_it = iter(plotly.colors.qualitative.Pastel2)

fig = go.Figure()
for it, ansatz in enumerate(ansaetze):
    main_color_sel = next(main_colors_it)
    sec_color_sel = rgb_to_rgba(next(sec_colors_it), 0.2)

    correlation_means = []
    for q in n_qubits:
        correlation_means.append(
            df[(df.ansatz == it) & (df.n_qubits == q)].coefficients_correlation_mean
        )
    correlation_means = np.array(correlation_means)
    correlation_means_mean = np.array(correlation_means).mean(axis=1)
    correlation_means_min = np.array(correlation_means).min(axis=1)
    correlation_means_max = np.array(correlation_means).max(axis=1)

    fig.add_trace(
        go.Scatter(
            x=n_qubits,
            y=correlation_means_mean,
            name=ansatz,
            mode="lines",
            line=dict(color=main_color_sel),
            marker=dict(color=main_color_sel),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=n_qubits,
            y=correlation_means_max,
            name=f"upper-{ansatz}",
            visible=True,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=n_qubits,
            y=correlation_means_min,
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
    title=f"Coefficient Correlation Mean for Different Ansaetze over Qubits",
    template="plotly_white",
    yaxis=dict(title="Coefficient Correlation Mean", type="log"),
    xaxis=dict(title="Qubits"),
)

hs = generate_hash(run_ids)
fig.write_image(f"results/coefficient_correlation_qubits_{hs}.pdf")
