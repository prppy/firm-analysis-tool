from sklearn.ensemble import IsolationForest

def detect_anomalies(df, feature_cols):
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )

    df["anomaly_score"] = iso.fit_predict(df[feature_cols])
    df["is_anomaly"] = (df["anomaly_score"] == -1).astype(int)

    return df
