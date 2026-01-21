from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_similarity(df, feature_cols):
    X = df[feature_cols].values
    sim_matrix = cosine_similarity(X)
    return sim_matrix

def get_top_peers(sim_matrix, index, top_n=5):
    scores = list(enumerate(sim_matrix[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores[1:top_n+1]
