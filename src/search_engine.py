import pandas as pd
from typing import Optional, Dict
from .data_loader import load_movies
from .query_parser import parse_query
from .metadata_filter import apply_filters
from .embeddings import get_or_create_embeddings, embed_query
from .vector_search import build_index, search

def search_movies(query: str, top_k: int = 5, preferences: Optional[Dict] = None):
    """Full search pipeline combining metadata filtering and semantic similarity.

    Returns a list of dictionaries with movie information and similarity score.
    """
    # Load data
    df = load_movies()

    # Parse query for filters
    filters = parse_query(query)

    # Apply metadata filters (genre, year, sorting) AND profile dealbreakers
    filtered_df = apply_filters(df, filters, preferences)
    if filtered_df.empty:
        return []

    # Get or compute embeddings for all movies
    all_embeddings = get_or_create_embeddings(df)

    # Subset embeddings to the filtered movies using original indices
    filtered_indices = filtered_df.index.tolist()
    filtered_embeddings = all_embeddings[filtered_indices]

    # Build FAISS index on filtered embeddings
    index = build_index(filtered_embeddings)

    # Embed the user query
    query_emb = embed_query(query)

    # Perform similarity search (get more results initially to re-rank)
    fetch_k = min(len(filtered_indices), max(top_k * 2, 20))
    distances, idxs = search(index, query_emb, top_k=fetch_k)

    results = []
    for dist, idx in zip(distances, idxs):
        # idx refers to position within filtered_embeddings
        original_idx = filtered_indices[idx]
        row = df.iloc[original_idx]
        
        # Base similarity score (since embeddings are normalized, dist is L2, lower is better. 
        # Convert to a pseudo-similarity where higher is better).
        # Normal range for L2 on normalized vectors is 0 to 2.
        base_sim = 1.0 - (float(dist) / 2.0)
        
        # Apply Profile Genre Weights
        genre = str(row["genre"]).lower()
        weight_boost = 0.0
        
        if preferences:
            # Map dashboard sliders (0 to 10) to a boost factor (-0.1 to +0.1)
            # Default weight is 5, which gives 0 boost.
            if "action" in genre:
                w = preferences.get("action_weight", 5)
                weight_boost += (w - 5) * 0.02
            if "comedy" in genre:
                w = preferences.get("comedy_weight", 5)
                weight_boost += (w - 5) * 0.02
            if "drama" in genre:
                w = preferences.get("drama_weight", 5)
                weight_boost += (w - 5) * 0.02
            if "sci-fi" in genre:
                w = preferences.get("scifi_weight", 5)
                weight_boost += (w - 5) * 0.02
                
        final_score = base_sim + weight_boost

        results.append({
            "title": row["title"],
            "genre": row["genre"],
            "release_year": int(row["release_year"]),
            "rating": float(row["rating"]),
            "description": row["description"],
            "score": final_score,
        })
        
    # Re-sort based on final weighted score (descending)
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    
    # Return top_k
    return results[:top_k]
