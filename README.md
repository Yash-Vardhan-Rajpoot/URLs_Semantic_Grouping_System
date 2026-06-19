# URLSense — URL Semantic grouping system

URLSense clusters and groups URLs by semantic similarity using embeddings and unsupervised clustering. The Streamlit app accepts a list of URLs, extracts page content, creates embeddings, performs agglomerative clustering, and returns labeled groups with a downloadable CSV.

## Features

- Parallel URL content extraction (Trafilatura / Newspaper3k / BeautifulSoup fallbacks)
- Sentence-transformers embeddings (BAAI/bge-large-en-v1.5)
- Cosine-similarity matrix and agglomerative clustering with automatic threshold selection
- Streamlit dashboard with cluster table, similarity matrix preview, and CSV export

## Quick Start

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1    # PowerShell on Windows
# or: .venv\Scripts\activate   # cmd.exe
```

2. Install dependencies:

```powershell
pip install streamlit sentence-transformers trafilatura newspaper3k beautifulsoup4 scikit-learn pandas numpy requests
```

3. Run the app:

```powershell
streamlit run app.py
```

Open the app in your browser, paste one URL per line, and click **Generate Clusters**.

## Project Layout

Top-level files:

- [app.py](app.py) — Streamlit UI and orchestration

Core modules (in `src/`):

- [src/scraper.py](src/scraper.py) — content extraction and parallel scraping
- [src/embedding.py](src/embedding.py) — loads SentenceTransformer and computes embeddings
- [src/clustering.py](src/clustering.py) — threshold selection and agglomerative clustering
- [src/category.py](src/category.py) — mapping cluster labels to human-friendly categories and result utilities
- [src/utils.py](src/utils.py) — text cleaning, slug extraction, CSV helper

## Notes & Recommendations

- The code uses `sentence-transformers` with model `BAAI/bge-large-en-v1.5`. Downloading and encoding this model can be large and may require considerable RAM or a GPU for reasonable speed.
- If you prefer reproducible installs, create a `requirements.txt` with the packages listed above.
- For production or large-scale runs, consider adding caching, rate limiting for scraping, and optional GPU acceleration.

## Future Improvements

1. HDBSCAN / density-based clustering option
2. Interactive cluster visualization (e.g., UMAP + Plotly)
3. GPU acceleration and batching for embeddings
4. Multi-language extraction and embedding support
5. Automatic cluster labeling using LLMs or heuristics
6. Real-time URL crawling and incremental updates

---

If you want, I can also generate a `requirements.txt` and add basic contributor/setup instructions.