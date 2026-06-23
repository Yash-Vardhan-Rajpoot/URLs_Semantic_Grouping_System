"""
Kaggle-style script: URL Semantic Grouping

Consolidates scraping, embedding, clustering, and category mapping
into a single class and prints only categories with their URLs.

Usage examples:
    python kaggle_notebook.py --urls "https://example.com,https://en.wikipedia.org/wiki/Natural_language_processing"
    python kaggle_notebook.py --urls-file input_urls.txt

On Kaggle, upload a file `input_urls.txt` with one URL per line and run the cell:
    !python kaggle_notebook.py --urls-file input_urls.txt
"""

from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, unquote
import argparse
import re
import os
import sys
from typing import List

import requests
from bs4 import BeautifulSoup
import trafilatura
from newspaper import Article

import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class URLSemanticGrouping:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", max_workers: int = 10):
        self.model_name = model_name
        self.max_workers = max_workers
        self.model = None

    # ----------------------
    # Utilities
    # ----------------------
    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _slug_text(self, url: str) -> str:
        path = urlparse(url).path
        slug = path.split("/")[-1]
        slug = unquote(slug)
        slug = slug.replace("-", " ").replace("_", " ")
        return slug.strip()

    # ----------------------
    # Scraping (fallbacks)
    # ----------------------
    def _extract_trafilatura(self, url: str):
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded, include_links=False, include_comments=False)
                if text and len(text) > 200:
                    return self._clean_text(text)
        except Exception:
            pass
        return None

    def _extract_newspaper(self, url: str):
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = (article.title or "") + " " + (article.text or "")
            if len(text) > 200:
                return self._clean_text(text)
        except Exception:
            pass
        return None

    def _extract_bs4(self, url: str):
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            title = soup.title.get_text(" ", strip=True) if soup.title else ""
            meta_description = ""
            meta = soup.find("meta", attrs={"name": "description"})
            if meta and meta.get("content"):
                meta_description = meta.get("content")

            headings = []
            for tag in ["h1", "h2", "h3"]:
                headings.extend([h.get_text(" ", strip=True) for h in soup.find_all(tag)])
            headings = " ".join(headings)

            body = soup.get_text(separator=" ", strip=True)
            body_words = body.split()
            body = " ".join(body_words[:300])

            slug = self._slug_text(url)
            final_text = f"TITLE:\n{title}\nMETA:\n{meta_description}\nHEADINGS:\n{headings}\nURL:\n{slug}\nCONTENT:\n{body}"
            return self._clean_text(final_text)
        except Exception:
            return None

    def _extract_content(self, url: str) -> str:
        text = self._extract_trafilatura(url)
        if text:
            return text
        text = self._extract_newspaper(url)
        if text:
            return text
        text = self._extract_bs4(url)
        if text:
            return text
        return self._slug_text(url)

    def scrape_urls(self, urls: List[str]) -> List[str]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            docs = list(ex.map(self._extract_content, urls))
        return docs

    # ----------------------
    # Embeddings
    # ----------------------
    def _load_model(self):
        if self.model is None:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers is not installed in the environment")
            print("Loading embedding model, this may take a while...")
            self.model = SentenceTransformer(self.model_name)

    def generate_embeddings(self, documents: List[str]):
        self._load_model()
        embeddings = self.model.encode(documents, normalize_embeddings=True, show_progress_bar=True)
        return embeddings

    def generate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        return cosine_similarity(embeddings)

    # ----------------------
    # Clustering
    # ----------------------
    def find_best_threshold(self, embeddings: np.ndarray) -> float:
        thresholds = np.arange(0.35, 0.65, 0.05)
        best_score = -1
        best_threshold = 0.45
        for threshold in thresholds:
            try:
                clusterer = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=threshold)
                labels = clusterer.fit_predict(embeddings)
                n_clusters = len(np.unique(labels))
                if n_clusters <= 1 or n_clusters >= len(labels):
                    continue
                score = silhouette_score(embeddings, labels, metric="cosine")
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            except Exception:
                pass
        return best_threshold

    def perform_clustering(self, embeddings: np.ndarray, sim_matrix: np.ndarray):
        n_urls = len(embeddings)
        if n_urls <= 10:
            threshold = 0.40
        else:
            threshold = self.find_best_threshold(embeddings)

        clusterer = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=threshold)
        labels = clusterer.fit_predict(embeddings)

        # Singleton detection
        next_cluster = (np.max(labels) + 1)
        for i in range(len(labels)):
            sims = np.delete(sim_matrix[i], i)
            max_similarity = np.max(sims)
            if max_similarity < 0.50:
                labels[i] = next_cluster
                next_cluster += 1

        # Re-number clusters
        unique_clusters = sorted(np.unique(labels))
        mapping = {old: new for new, old in enumerate(unique_clusters)}
        labels = np.array([mapping[label] for label in labels])
        return labels, threshold

    # ----------------------
    # Categories & Output
    # ----------------------
    def create_result_dataframe(self, urls: List[str], labels: np.ndarray) -> pd.DataFrame:
        cluster_names = {cluster: f"Category {idx+1}" for idx, cluster in enumerate(sorted(np.unique(labels)))}
        categories = [cluster_names[label] for label in labels]
        result = pd.DataFrame({"URL": urls, "Cluster": labels, "Category": categories})
        return result

    def print_category_groups(self, result: pd.DataFrame):
        for category in sorted(result["Category"].unique()):
            print(f"{category} ({len(result[result['Category'] == category])} URLs)")
            for url in result[result["Category"] == category]["URL"]:
                print(" •", url)
            print("")

    # ----------------------
    # Orchestration
    # ----------------------
    def run(self, urls: List[str]):
        urls = [u.strip() for u in urls if u and u.strip()]
        if len(urls) < 1:
            print("No URLs provided.")
            return

        print(f"Scraping {len(urls)} URLs...")
        docs = self.scrape_urls(urls)
        docs = [d[:5000] if d else "" for d in docs]

        print("Generating embeddings...")
        embeddings = self.generate_embeddings(docs)

        print("Calculating similarity matrix...")
        sim_matrix = self.generate_similarity_matrix(embeddings)

        print("Performing clustering...")
        labels, threshold = self.perform_clustering(embeddings, sim_matrix)

        result = self.create_result_dataframe(urls, labels)

        # Print only categories with URLs (no matrices or extra outputs)
        self.print_category_groups(result)


def _read_urls_from_file(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", help="Comma separated URLs", default=None)
    parser.add_argument("--urls-file", help="Plain text file with one URL per line", default=None)
    args = parser.parse_args()

    urls = []
    if args.urls_file:
        urls = _read_urls_from_file(args.urls_file)
    elif args.urls:
        urls = [u.strip() for u in args.urls.split(",") if u.strip()]
    else:
        # Default sample URLs for quick runs
        urls = [
            "https://example.com",
            "https://en.wikipedia.org/wiki/Natural_language_processing",
        ]

    runner = URLSemanticGrouping()
    try:
        runner.run(urls)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
