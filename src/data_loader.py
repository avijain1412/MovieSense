import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'movies.csv')

def load_movies() -> pd.DataFrame:
    """Load movies CSV into a DataFrame.

    Returns:
        pd.DataFrame with columns: movie_id, title, genre, release_year, description, rating
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Movie dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    # Ensure proper types
    df['release_year'] = df['release_year'].astype(int)
    df['rating'] = df['rating'].astype(float)
    return df
