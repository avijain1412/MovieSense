# MovieSense 📽️

**MovieSense** is a lightweight demo that combines **semantic vector search** with **metadata filtering** to let users find movies that match the *meaning* of their natural‑language queries while respecting explicit constraints like genre or release year.

---

## 🚀 Quick Overview
- **Goal**: Return movies that are semantically relevant to a query such as "funny space movies" even if the exact words are missing, while allowing filters like `genre: comedy` or `year: before 2010`.
- **Approach**:
  1. Parse the query to extract optional filters (genre, year, etc.).
  2. Filter the CSV dataset using Pandas.
  3. Generate embeddings with a **Sentence‑Transformer** (`all‑MiniLM‑L6‑v2`).
  4. Build a **FAISS‑CPU** index for fast similarity search.
  5. Perform a vector similarity lookup for the user query.
  6. Display the top‑k results via **Streamlit**.

---

## 🛠️ Tech Stack
- **Python** – core language
- **Pandas** – CSV handling & filtering
- **Sentence‑Transformers** – text → vector embeddings
- **FAISS‑CPU** – efficient similarity search
- **Streamlit** – simple interactive UI

---

## 📂 Project Structure
```
MovieSense 2/
│   app.py               # Streamlit UI entry point
│   requirements.txt
│   README.md            # (this file)
│   .gitignore
│
├── data/
│   └── movies.csv       # Sample movie dataset
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py   # Load CSV into a DataFrame
│   ├── query_parser.py  # Rule‑based extraction of filters
│   ├── metadata_filter.py # Apply genre/year filters & sorting
│   ├── embeddings.py    # Compute / load embeddings
│   ├── vector_search.py # FAISS build & search helpers
│   └── search_engine.py # Orchestrates the full pipeline
│
└── utils/
    └── __init__.py
```

---

## ⚙️ Installation
```bash
# Create a virtual environment (recommended)
python3 -m venv env
source env/bin/activate   # Windows: env\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Running the App
```bash
streamlit run app.py
```
The UI opens in your browser. Type a natural‑language query (e.g., `"funny comedy movies before 2010"`) and press **Search**.

---

## 💡 Example Queries
- `I want a comedy movie`
- `action movies after 2015`
- `anime movies before 2020`
- `funny movies before 2010`
- `latest comedy movie before 2020`
- `earliest action movie`
- `science fiction movies after 2010`
- `no matching movies query`

---

## ⚠️ Limitations
- The query parser is rule‑based; only a limited set of genres and simple year expressions are supported.
- Embeddings are computed on‑the‑fly the first run and cached to `data/movie_embeddings.npy`.
- No external APIs are required; the OpenAI API is **not** used.
- The dataset is tiny and intended for demonstration only.

---

## 🔮 Future Improvements
- Expand the genre list and replace the rule‑based parser with a lightweight language model.
- Add pagination for result lists.
- Allow users to upload their own CSV datasets.
- Persist the FAISS index to avoid recomputation on every start.

---

## 📄 License
This project is licensed under the MIT License. See the `LICENSE` file for details.
