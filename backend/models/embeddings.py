from sentence_transformers import SentenceTransformer
from functools import lru_cache

@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer("BAAI/bge-small-en-v1.5")