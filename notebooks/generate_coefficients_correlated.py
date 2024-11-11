import mlflow
import plotly
from plotly.subplots import make_subplots
import re
import json
import pandas as pd
import plotly.io as pio
from notebooks.runs import run_ids, experiment_id

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

for q in qubits:
    fig = make_subplots(rows=1, cols=n_ansaetze, subplot_titles=ansaetze)

    it = 1
    for index, row in global_df[global_df.qubits == q].iterrows():
        client = mlflow.tracking.MlflowClient()

        sub_fig_path = client.download_artifacts(
            row.run_id, f"coefficients_correlated.html", "./"
        )
        sub_fig = read_from_html(sub_fig_path)
        sub_fig_trace = sub_fig.data[0]
        sub_fig_trace.update(coloraxis=f"coloraxis")

        fig.add_trace(sub_fig_trace, row=1, col=it)
        fig.update_xaxes(dict(title="Coefficients"), row=1, col=it)
        fig.update_yaxes(
            dict(
                title="Coefficients" if it == 1 else "",
                autorange="reversed",
                scaleanchor="x",
            ),
            row=1,
            col=it,
        )
        it = it + 1

    fig.update_layout(
        title_text=f"Correlation of Coefficients for Different Ansaetze ({q} Qubits)",
        template="plotly_white",
        height=400,
        width=300 * it,
        coloraxis={"colorscale": "Bluyl"},
    )

    fig.write_image(f"new_coefficients_correlated_q{q}.pdf")
