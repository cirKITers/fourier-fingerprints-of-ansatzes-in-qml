import mlflow
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import json
import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.express as px

pio.kaleido.scope.mathjax = None

experiment_id = "446249578927817437"
run_ids = [
    "34cc3379811f497790de5cb84dd3e3af",
    "ca422eaec0ef498992af889083155a2c",
    "e8133ab587014ab8893a7e570ba2465c",
    "f7c69c0caebe4e6c811d2195c66237a5",
    "d96866eda07a4260876e5dfbd369bf61",
    "cc2d405607634d2c9237a996d29b9219",
]

run_ids = [
    "fb2e0302178241038c80dbcb6a53b1f2",
    "d471f2a89bbb4d82870cb730a0582639",
    "2645b5f807614900987a4c773f483c83",
    "e7427328b69e4adb8e2477570fc75542",
    "536c655447874e90a427501a104d7162",
    "5c8086065a814b6b965d2d9001ca6ef2",
]

# run_id_condition = "'" + "','".join(run_ids) + "'"
# complex_filter = f"""
#     attributes.run_id IN ({run_id_condition})
#     """

# runs = mlflow.search_runs(
#     experiment_ids=[experiment_id],
#     filter_string=complex_filter,
# )


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
    ansaetze.append(ansatz)
# ----------------------------------


fig = make_subplots(rows=1, cols=len(run_ids), subplot_titles=ansaetze)

for it, run_id in enumerate(run_ids):
    client = mlflow.tracking.MlflowClient()

    sub_fig_path = client.download_artifacts(
        run_id, f"coefficients_correlated.html", "./"
    )
    sub_fig = read_from_html(sub_fig_path)
    sub_fig_trace = sub_fig.data[0]
    sub_fig_trace.update(coloraxis=f"coloraxis")

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
    title_text="Correlation of Coefficients for Different Ansaetze",
    template="plotly_white",
    height=400,
    width=300 * len(run_ids),
)

fig.write_image("coefficients_correlated.pdf")

# ------------------------

loss_precision = 1e3
df = pd.DataFrame()

for it, run_id in enumerate(run_ids):
    client = mlflow.tracking.MlflowClient()
    mse_hist = client.get_metric_history(run_id, "mse")
    df[ansaetze[it]] = [
        np.trunc(entity.value * loss_precision) / loss_precision for entity in mse_hist
    ]

fig = go.Figure(
    data=[go.Scatter(x=df.index, y=df[ansatz], name=ansatz) for ansatz in ansaetze]
)

fig.update_layout(
    title="Loss for Different Ansaetze",
    template="plotly_white",
    yaxis=dict(title="MSE", type="log", range=[np.log(1 / loss_precision), np.log(1)]),
    xaxis=dict(title="Epochs", type="log"),
)

fig.write_image("mse.pdf")

fig = make_subplots(rows=1, cols=len(run_ids), subplot_titles=ansaetze)

for it, run_id in enumerate(run_ids):
    client = mlflow.tracking.MlflowClient()

    sub_fig_path = client.download_artifacts(
        run_id, f"coefficients_correlated.html", "./"
    )
    sub_fig = read_from_html(sub_fig_path)
    sub_fig_trace = sub_fig.data[0]
    sub_fig_trace.update(coloraxis=f"coloraxis")

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
    title_text="Correlation of Coefficients for Different Ansaetze",
    template="plotly_white",
    height=400,
    width=300 * len(run_ids),
)

fig.write_image("coefficients_correlated.pdf")

# ------------------------

loss_precision = 1e3
df = pd.DataFrame()

for it, run_id in enumerate(run_ids):
    client = mlflow.tracking.MlflowClient()
    mse_hist = client.get_metric_history(run_id, "mse")
    df[ansaetze[it]] = [
        np.trunc(entity.value * loss_precision) / loss_precision for entity in mse_hist
    ]

fig = go.Figure(
    data=[go.Scatter(x=df.index, y=df[ansatz], name=ansatz) for ansatz in ansaetze]
)

fig.update_layout(
    title="Loss for Different Ansaetze",
    template="plotly_white",
    yaxis=dict(title="MSE", type="log", range=[np.log(1 / loss_precision), np.log(1)]),
    xaxis=dict(title="Epochs", type="log"),
)

fig.write_image("mse.pdf")

# ------------------------

df = pd.DataFrame(
    columns=[
        "ansatz",
        "mse",
        "coefficients_correlation_mean",
        # "coefficients_correlation_variance",
    ]
)

for it, run_id in enumerate(run_ids):
    client = mlflow.tracking.MlflowClient()
    client.get_run(run_id).data
    df.loc[it, "ansatz"] = client.get_run(run_id).data.params["circuit_type"]
    df.loc[it, "mse"] = client.get_run(run_id).data.metrics["mse"]
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
            line=dict(color=df.index, colorscale=px.colors.qualitative.Dark2),
            dimensions=list(
                [
                    dict(
                        label="ansatz",
                        values=df.index,
                        tickvals=df.index,
                        ticktext=df["ansatz"],
                    ),
                    # dict(
                    #     label="coefficients_correlation_variance",
                    #     values=df["coefficients_correlation_variance"],
                    # ),
                    dict(
                        label="coefficients_correlation_mean",
                        values=df["coefficients_correlation_mean"],
                    ),
                    dict(label="mse", values=df["mse"]),
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

fig.write_image("parcoords.pdf")
