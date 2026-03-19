from __future__ import annotations

import io
import json
import random
import time
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from pandas.api.types import is_numeric_dtype

st.set_page_config(page_title="LAB2 App")
sns.set_theme(style="whitegrid")

DEFAULT_DATASET_URL = "https://cdn.jsdelivr.net/npm/vega-datasets@latest/data/cars.json"


def init_state() -> None:
    defaults = {
        "variant": None,
        "start_time": None,
        "records": [],
        "active_settings": None,
        "review_score": 5,
        "review_feedback": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_data(show_spinner=False)
def load_dataset(file_bytes: bytes | None = None, filename: str | None = None) -> pd.DataFrame:
    if file_bytes is None:
        with urlopen(DEFAULT_DATASET_URL, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
        return pd.json_normalize(data)

    suffix = Path(filename).suffix.lower()
    buffer = io.BytesIO(file_bytes)

    if suffix in {".csv", ".txt"}:
        return pd.read_csv(buffer)
    if suffix == ".tsv":
        return pd.read_csv(buffer, sep="\t")
    if suffix == ".json":
        data = json.loads(file_bytes.decode("utf-8"))
        if isinstance(data, list):
            return pd.json_normalize(data)
        if isinstance(data, dict):
            for key in ["values", "data", "records", "items", "rows"]:
                if isinstance(data.get(key), list):
                    return pd.json_normalize(data[key])
            return pd.DataFrame(data)

    raise ValueError("Please upload a CSV, TSV, TXT, or JSON file.")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(col).strip().replace(" ", "_") for col in df.columns]
    return df


def get_column_options(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = [col for col in df.columns if is_numeric_dtype(df[col])]
    categorical_cols = [
        col for col in df.columns
        if col not in numeric_cols and 1 < df[col].nunique(dropna=True) <= 20
    ]
    return numeric_cols, categorical_cols


def draw_chart(df: pd.DataFrame, metric: str, category: str, variant: str):
    fig, ax = plt.subplots(figsize=(8, 5))

    if category == "(None)":
        chart_df = df[[metric]].dropna()
        if variant == "A":
            sns.histplot(chart_df, x=metric, kde=True, ax=ax)
        else:
            sns.boxplot(chart_df, x=metric, ax=ax)
    else:
        chart_df = df[[metric, category]].dropna()
        order = chart_df.groupby(category)[metric].mean().sort_values(ascending=False).index.tolist()
        if variant == "A":
            sns.barplot(
                chart_df,
                x=category,
                y=metric,
                estimator="mean",
                errorbar=None,
                order=order,
                ax=ax,
            )
        else:
            sns.boxplot(chart_df, x=category, y=metric, order=order, ax=ax)
        ax.tick_params(axis="x", rotation=20)

    ax.set_title(f"Chart {variant}")
    fig.tight_layout()
    return fig


def record_answer(answer: str, dataset_name: str, question: str) -> None:
    start_time = st.session_state.start_time
    elapsed = round(time.time() - start_time, 2) if start_time is not None else None

    if answer == "Yes":
        st.balloons()

    st.session_state.records.append({
        "dataset": dataset_name,
        "question": question,
        "chart": st.session_state.variant,
        "answer": answer,
        "time_seconds": elapsed,
    })

    st.session_state.start_time = None
    st.session_state.variant = None


def submit_review() -> None:
    score = st.session_state.review_score
    if 1 <= score <= 5:
        st.session_state.review_feedback = "snow"
    else:
        st.session_state.review_feedback = "balloons"


def render_review_feedback() -> None:
    feedback = st.session_state.review_feedback

    if feedback == "snow":
        st.snow()
        st.session_state.review_feedback = None
    elif feedback == "balloons":
        st.balloons()
        st.session_state.review_feedback = None


def main() -> None:
    init_state()

    title_col, review_col = st.columns([8, 1])

    with title_col:
        st.title("LAB2")

    with review_col:
        with st.popover("Review"):
            st.slider("Rate this app", 1, 10, key="review_score")
            st.button("Submit", use_container_width=True, on_click=submit_review)

    render_review_feedback()

    st.write("Choose a dataset")

    if st.radio("Dataset source", ["Use default dataset", "Upload my own dataset"]) == "Use default dataset":
        dataset_name, df = "cars", load_dataset()
    else:
        uploaded_file = st.file_uploader(
            "Upload CSV, TSV, TXT, or JSON",
            type=["csv", "tsv", "txt", "json"],
        )
        if not uploaded_file:
            st.stop()
        dataset_name = Path(uploaded_file.name).stem
        df = load_dataset(uploaded_file.getvalue(), uploaded_file.name)

    df = clean_dataframe(df)
    numeric_cols, categorical_cols = get_column_options(df)

    if not numeric_cols:
        st.error("Your dataset needs at least one numeric column.")
        st.stop()

    metric = st.selectbox(
        "Numeric variable",
        numeric_cols,
        index=numeric_cols.index("Miles_per_Gallon") if "Miles_per_Gallon" in numeric_cols else 0,
    )

    category_options = ["(None)"] + categorical_cols
    category = st.selectbox(
        "Grouping variable",
        category_options,
        index=category_options.index("Origin") if "Origin" in category_options else 0,
    )

    question = (
        f"What does the distribution of {metric} look like in the {dataset_name} dataset?"
        if category == "(None)"
        else f"Which {category} has the highest average {metric} in the {dataset_name} dataset?"
    )

    st.subheader("Business question")
    st.info(question)

    settings_key = (dataset_name, metric, category)
    if st.session_state.active_settings and st.session_state.active_settings != settings_key:
        st.session_state.variant = None
        st.session_state.start_time = None
        st.session_state.active_settings = None

    if st.button("Show one chart at random", type="primary"):
        st.session_state.variant = random.choice(["A", "B"])
        st.session_state.start_time = time.time()
        st.session_state.active_settings = settings_key

    if st.session_state.variant:
        st.pyplot(draw_chart(df, metric, category, st.session_state.variant), clear_figure=True)
        st.write(f"Can you answer the question using graph {st.session_state.variant}?")

        col1, col2 = st.columns(2)
        col1.button(
            "Yes",
            use_container_width=True,
            on_click=record_answer,
            args=("Yes", dataset_name, question),
        )
        col2.button(
            "No",
            use_container_width=True,
            on_click=record_answer,
            args=("No", dataset_name, question),
        )

    if st.session_state.records:
        last = st.session_state.records[-1]
        time_label = f"{last['time_seconds']}s" if last["time_seconds"] is not None else "N/A"
        st.success(f"Last result: Chart {last['chart']} | Answer: {last['answer']} | Time: {time_label}")
        st.dataframe(pd.DataFrame(st.session_state.records), use_container_width=True)


if __name__ == "__main__":
    main()
