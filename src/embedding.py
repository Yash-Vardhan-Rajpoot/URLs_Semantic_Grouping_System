# ==========================================================
# embedding.py
# Embedding Generation Module
# ==========================================================

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# LOAD MODEL
# ==========================================================

print("Loading Embedding Model...")

model = SentenceTransformer(
    "BAAI/bge-large-en-v1.5"
)

print("Model Loaded Successfully!")


# ==========================================================
# GENERATE EMBEDDINGS
# ==========================================================

def generate_embeddings(documents):

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings


# ==========================================================
# SIMILARITY MATRIX
# ==========================================================

def generate_similarity_matrix(
    embeddings
):

    return cosine_similarity(
        embeddings
    )