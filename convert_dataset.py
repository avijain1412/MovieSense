import pandas as pd

# Load Kaggle dataset
df = pd.read_csv("imdb_top_1000.csv")

# Convert to MovieSense format
movies = pd.DataFrame({
    "movie_id": range(1, len(df) + 1),
    "title": df["Series_Title"],
    "genre": df["Genre"],
    "release_year": pd.to_numeric(df["Released_Year"], errors="coerce"),
    "description": df["Overview"],
    "rating": df["IMDB_Rating"]
})

# Remove rows with missing important information
movies = movies.dropna(
    subset=["title", "genre", "release_year", "description"]
)

# Make release year an integer
movies["release_year"] = movies["release_year"].astype(int)

# Save in the format MovieSense expects
movies.to_csv("data/movies.csv", index=False)

print(f"Saved {len(movies)} movies to data/movies.csv")
