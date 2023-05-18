"""
Aggregate evaluation results from a directory of evaluation results.
"""
import io
import os
import json
import base64
import urllib
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from glob import glob
from io import BytesIO

sns.set_theme()

parser = argparse.ArgumentParser()
parser.add_argument("--dir", required=True)
# test.jsonl.head_only.eval.md
parser.add_argument("--filename", required=True)
parser.add_argument("--eval-id", required=True)  # headonly OR span OR coref
args = parser.parse_args()

eval_files = glob(os.path.join(args.dir, "*", args.filename), recursive=True)
print(json.dumps(vars(args), indent=4))
print(f"Found {len(eval_files)} eval files\n")


def parse_md_table_to_pandas_df(md_table_str: str):
    f = io.StringIO(md_table_str)
    df = pd.read_csv(f, sep="|", header=0, index_col=1, skipinitialspace=True)
    # drop the left-most and right-most null columns
    df = df.dropna(axis=1, how="all")
    df = df.iloc[1:]  # Drop the header underline row
    return df


# 1. Aggregate all eval files
eval_id_to_metric_table = {}
eval_id_to_stats = {}
for eval_file in eval_files:
    eval_id = os.path.basename(os.path.dirname(eval_file))
    # print(f"# Eval `{eval_id}`\n")

    stats = []
    metric_table = []
    _collect_stats = False
    _collect_metric = False
    with open(eval_file, "r") as f:
        for line in f:
            if line.startswith("## Prediction Stats"):
                _collect_stats = True
            if line.startswith("## Performance Stats"):
                _collect_stats = False
                _collect_metric = True

            # indent header level for markdown
            if line.startswith("#"):
                line = "#" + line

            if _collect_stats and line.startswith("|"):
                stats.append(line)
            if _collect_metric and line.startswith("|"):
                metric_table.append(line)

    try:
        eval_id_to_stats[eval_id] = parse_md_table_to_pandas_df("".join(stats))
    except Exception as e:
        raise Exception(f"Failed to parse stats for {eval_file}")
    try:
        eval_id_to_metric_table[eval_id] = parse_md_table_to_pandas_df(
            "".join(metric_table))
    except Exception as e:
        raise Exception(f"Error parsing metric table for {eval_file}")

# 2. Aggregate all eval metrics


def eval_id_to_info(eval_id):
    # get rid of extra suffix
    repeat_id = ""
    if "repeat" in eval_id or "incontextknn" in eval_id:
        repeat_id = eval_id.split("-")[-1]
        eval_id = "-".join(eval_id.split("-")[:-1])
    n, temp = eval_id.split("-")[-2:]
    version = "-".join(eval_id.split("-")[:-2])
    n = int(n[1:])
    temp = float(temp[1:])

    if "shot" in version.split("-")[-1]:
        k_shot = int(version.split("-")[-1].replace("shot", ""))
        # version = "-".join(version.split("-")[:-1])
    else:
        k_shot = 0

    return {
        "version": version,
        "n": n,
        "temp": temp,
        "k_shot": k_shot,
        "repeat_id": repeat_id,
    }


# 2.1 Save stats
for eval_id, stats in eval_id_to_stats.items():
    for k, v in eval_id_to_info(eval_id).items():
        stats.loc[k] = v
    print(f"## Stats for `{eval_id}`\n")
    print(stats)
stats = pd.concat([d.T for d in eval_id_to_stats.values()])
stats.columns = list(map(lambda x: x.strip(), stats.columns))
stats.reset_index(inplace=True, drop=True)

# 2.2 Aggregate metrics and print out as md
for eval_id, metric_table in eval_id_to_metric_table.items():
    for k, v in eval_id_to_info(eval_id).items():
        metric_table[k] = v

metric_table = pd.concat(eval_id_to_metric_table.values())
metric_table.columns = list(map(lambda x: x.strip(), metric_table.columns))
metric_table = metric_table.reset_index().rename(columns={"index": "Metric"})

# 2.3 Aggregate metric and stats into one table


def _remove_kshots_in_version(version: str):
    return version if "shot" not in version.split("-")[-1] else "-".join(version.split("-")[:-1])


aggregated_table = metric_table.copy()
aggregated_table["version"] = aggregated_table["version"].apply(
    _remove_kshots_in_version)
stats["version"] = stats["version"].apply(_remove_kshots_in_version)
# joint both table on version, n, temp, k_shot
aggregated_table = aggregated_table.merge(
    stats, on=["version", "n", "temp", "k_shot", "repeat_id"])
aggregated_table.to_csv(os.path.join(
    args.dir, f"eval-{args.eval_id}-metrics.csv"), index=False)
