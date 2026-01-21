import streamlit as st

import sys
sys.path.append("./src")

import pandas as pd

from preprocessing import load_data, basic_cleaning, coerce_numeric_columns, handle_missing_values, handle_range_values
from feature_engineering import create_features, log_transform
from clustering import run_kmeans
from similarity import compute_similarity
from anomaly import detect_anomalies
from config import NUMERIC_COLS, RANGE_COLS

# TODO: PARAMA TO DO UP THE WEBSITE AFTER ALL STEPS ARE FINISHED 
st.set_page_config(page_title="AI Company Intelligence", layout="wide")

st.title("AI-Driven Company Intelligence")
st.markdown(
    """
    This prototype demonstrates data-driven company segmentation,
    peer comparison, and interpretable insights.
    """
)

@st.cache_data
def load_and_prepare_data():
    df = load_data("./data/champions_group_data.xlsx")
    df = basic_cleaning(df)
    df = coerce_numeric_columns(df, NUMERIC_COLS)
    df = handle_missing_values(df, NUMERIC_COLS)
    df = handle_range_values(df, RANGE_COLS)
    df = create_features(df)
    return df

df = load_and_prepare_data()

st.header("Company Explorer")

company_col = "Company Sites" 
company = st.selectbox(
    "Select a company",
    sorted(df[company_col].dropna().unique())
)

company_df = df[df[company_col] == company]

st.subheader("Company Snapshot")
st.dataframe(company_df.T)


st.header("Company Segmentation")

k = st.slider("Number of clusters", 3, 10, 5)

clustered_df = run_kmeans(df, k=k)

st.write("Cluster distribution")
st.bar_chart(clustered_df["cluster"].value_counts())

st.dataframe(clustered_df[[company_col, "cluster"]].head())

st.header("Peer Comparison")

top_n = st.slider("Number of peers", 3, 10, 5)

peers = compute_similarity(
    df,
    target_company=company,
    top_n=top_n
)

st.subheader("Most Similar Companies")
st.dataframe(peers)

from insights import generate_insights

st.header("AI-Generated Insights")

insights = generate_insights(company_df, peers)

st.markdown(insights)