import re
from typing import Optional, Dict

# Simple rule‑based query parser for movie searches

def parse_query(query: str) -> Dict:
    """Extract structured filters from a natural‑language query.

    The returned dictionary may contain the following keys:
        genre (str) – detected genre name (case‑insensitive)
        year (int) – numeric year if found
        year_operator (str) – "before", "after", or None
        sort_order (str) – "desc" for latest, "asc" for earliest, or None
    """
    result: Dict[str, Optional[object]] = {
        "genre": None,
        "year": None,
        "year_operator": None,
        "sort_order": None,
    }

    lowered = query.lower()

    # ----- genre detection -----
    # Simple list based on our dataset – extend as needed
    known_genres = ["sci-fi", "science fiction", "comedy", "action", "anime", "drama", "fantasy", "thriller"]
    for g in known_genres:
        if g in lowered:
            # normalise to a consistent representation (capitalised first letter)
            result["genre"] = g.replace("science fiction", "Sci-Fi").title() if g != "sci-fi" else "Sci-Fi"
            break

    # ----- year and operator detection -----
    # Look for patterns like "after 2015", "before 2020", "2010", "movies from 1999"
    year_match = re.search(r"(\d{4})", lowered)
    if year_match:
        result["year"] = int(year_match.group(1))
        # Determine operator based on surrounding words
        before = re.search(r"before\s+" + str(result["year"]), lowered)
        after = re.search(r"after\s+" + str(result["year"]), lowered)
        if before:
            result["year_operator"] = "before"
        elif after:
            result["year_operator"] = "after"
        else:
            # No explicit operator – treat as exact match (handled later as equality)
            result["year_operator"] = None

    # ----- sort order detection (latest / earliest) -----
    if "latest" in lowered or "most recent" in lowered:
        result["sort_order"] = "desc"
    elif "earliest" in lowered or "oldest" in lowered:
        result["sort_order"] = "asc"

    return result
