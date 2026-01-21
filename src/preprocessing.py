import pandas as pd
import numpy as np
import re

def load_data(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, engine="openpyxl")
    elif path.lower().endswith(".csv"):
        return pd.read_csv(path, encoding="utf-8")
    else:
        raise ValueError("Unsupported file format")

def basic_cleaning(df):
    # Standardize column names
    df.columns = df.columns.str.strip()

    # Standardise empty / unknown values
    df = df.replace(
        ["", " ", "Unknown", "UNKNOWN", "N/A", "NA"],
        np.nan
    )

    return df

def handle_missing_values(df, numeric_cols):
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[f"has_{col.lower().replace(' ', '_')}"] = df[col].notna().astype(int)
        df[col] = df[col].fillna(df[col].median())
    return df

def coerce_numeric_columns(df, numeric_cols):
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def handle_range_values(df, range_cols):
    for col in range_cols:
        df[col] = df[col].apply(_parse_range)
    return df

def _parse_range(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)):
        return value

    match = re.findall(r"\d+", str(value))
    if len(match) == 2:
        return (int(match[0]) + int(match[1])) / 2
    elif len(match) == 1:
        return int(match[0])

    return np.nan
