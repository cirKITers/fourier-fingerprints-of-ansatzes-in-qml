import plotly
import re
import json
import hashlib

def generate_hash(run_ids):
    hs = hashlib.md5(repr(run_ids).encode("utf-8")).hexdigest()
    return hs

def read_from_html(path):
    with open(path) as f:
        html = f.read()
    call_arg_str = re.findall(r"Plotly\.newPlot\((.*)\)", html[-(2**16) :])[0]
    call_args = json.loads(f"[{call_arg_str}]")
    plotly_json = {"data": call_args[1], "layout": call_args[2]}
    return plotly.io.from_json(json.dumps(plotly_json))