from glob import glob
import pandas as pd
import streamlit as st
import argparse
import os

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

# https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters", key="-".join(df.columns))

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(
                        map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(
                        str).str.contains(user_text_input)]

    return df


parser = argparse.ArgumentParser()
parser.add_argument('--eval-folder', type=str,
                    default='data/extraction/ace/inferred')
args = parser.parse_args()

st.set_page_config(layout="wide")
st.title("Model Evaluation Aggregation Visualizer")

model_selection = st.selectbox(
    'Visualize which model?',
    glob(args.eval_folder + "/*/*"),
)
viz_selection = st.selectbox(
    'Visualize which eval type?',
    ["headonly", "span", "coref"]
)

st.write(f"Visualizing `{model_selection}` `{viz_selection}`")

# =================================================


def _remove_kshots_in_version(version: str):
    return version if "shot" not in version.split("-")[-1] else "-".join(version.split("-")[:-1])


st.header("Aggregated Metrics")
metrics = pd.read_csv(
    os.path.join(
        model_selection,
        f"eval-{viz_selection}-metrics.csv"
    )
)


def _select_columns(metrics):
    # filter columns by multi-select
    _left_columns = ["version", "n", "temp", "k_shot"]
    _data_columns = [c for c in metrics.columns if c not in _left_columns]
    _data_columns = st.multiselect(
        "Select columns to display",
        _data_columns,
        default=_data_columns,
        key="column_selector"
    )
    df = metrics[_left_columns + _data_columns]
    return df


df = _select_columns(metrics)
# round all float to 2 decimal places
df = df.round(2)
# df = df.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x)

# visualize with heatmap style using pandas
st.dataframe(
    filter_dataframe(df).style.background_gradient(
        # subset=_right_columns,
        cmap='Blues'
    ))
