import pandas as pd
from typing import Dict, Optional


def apply_filters(df: pd.DataFrame, filters: Dict, preferences: Optional[Dict] = None) -> pd.DataFrame:
    """Apply genre and year filters to the DataFrame.

    Args:
        df: DataFrame containing movie data.
        filters: Dictionary from ``parse_query`` with keys ``genre``, ``year``,
            ``year_operator`` and ``sort_order``.
        preferences: Optional dictionary of profile dashboard preferences.
    Returns:
        Filtered DataFrame.
    """
    filtered = df.copy()

    # Apply Dealbreakers from Profile Dashboard first
    if preferences and "dealbreakers" in preferences:
        dealbreakers = [d.lower() for d in preferences["dealbreakers"]]
        if dealbreakers:
            # Exclude movies where genre exactly matches or contains the dealbreaker
            # In our simple dataset, genre is a single string.
            filtered = filtered[~filtered["genre"].str.lower().isin(dealbreakers)]
            # Note: A more complex dataset might have comma-separated genres, but we'll stick to exact match for now.

    # Genre filter (case‑insensitive)
    genre = filters.get("genre")
    if genre:
        filtered = filtered[filtered["genre"].str.lower() == genre.lower()]

    # Year filter
    year = filters.get("year")
    op = filters.get("year_operator")
    if year is not None and op:
        if op == "before":
            filtered = filtered[filtered["release_year"] < year]
        elif op == "after":
            filtered = filtered[filtered["release_year"] > year]
    elif year is not None:
        # Exact match if no operator
        filtered = filtered[filtered["release_year"] == year]

    # Sorting for latest / earliest
    sort_order = filters.get("sort_order")
    if sort_order == "desc":
        filtered = filtered.sort_values(by="release_year", ascending=False)
    elif sort_order == "asc":
        filtered = filtered.sort_values(by="release_year", ascending=True)

    return filtered  # Keep original indices for embedding lookup
