from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pandas as pd

# Path to store embeddings
EMBEDDING_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'movie_embeddings.npy')
MODEL_NAME = "all-MiniLM-L6-v2"

def load_model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer model."""
    return SentenceTransformer(MODEL_NAME)

def compute_embeddings(df: pd.DataFrame) -> np.ndarray:
    """Compute embeddings for each movie based on title, genre and description.

    Returns:
        np.ndarray of shape (num_movies, embedding_dim)
    """
    model = load_model()
    texts = (
        df["title"].astype(str) + ". " +
        df["genre"].astype(str) + ". " +
        df["description"].astype(str)
    ).tolist()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings

def get_or_create_embeddings(df: pd.DataFrame) -> np.ndarray:
    """Load embeddings from disk if present, otherwise compute and save them."""
    if os.path.exists(EMBEDDING_PATH):
        return np.load(EMBEDDING_PATH)
    embeddings = compute_embeddings(df)
    os.makedirs(os.path.dirname(EMBEDDING_PATH), exist_ok=True)
    np.save(EMBEDDING_PATH, embeddings)
    return embeddings

def embed_query(query: str) -> np.ndarray:
    """Generate a normalized embedding for a user query."""
    model = load_model()
    emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return emb[0]
