from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_kmeans(df, feature_cols, k=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)

    return labels, score, scaler, kmeans
