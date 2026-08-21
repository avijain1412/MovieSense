import faiss
import numpy as np

def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Create a FAISS index for the given embeddings.

    The embeddings are assumed to be L2‑normalized, so inner product works as cosine similarity.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product
    index.add(embeddings)
    return index

def search(index: faiss.Index, query_emb: np.ndarray, top_k: int = 5):
    """Search the FAISS index for nearest neighbours.

    Returns:
        distances (np.ndarray): similarity scores (higher is more similar)
        indices (np.ndarray): corresponding indices in the original embeddings array
    """
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    distances, indices = index.search(query_emb, top_k)
    return distances[0], indices[0]
