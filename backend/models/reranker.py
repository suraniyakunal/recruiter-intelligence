from sentence_transformers import CrossEncoder
from functools import lru_cache
import numpy as np
from typing import List, Dict

@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder("BAAI/bge-reranker-base")

def rerank_candidates(
    jd_text: str,
    candidates: List[Dict],   # each dict must have at least "text" key (e.g., raw_text)
    top_k: int = 300
) -> List[Dict]:
    """
    candidates: list of dicts from retrieval, with "score" (hybrid score) and "text".
    Returns list with added "reranker_score" and new "final_score" blended.
    """
    reranker = get_reranker()
    pairs = [[jd_text, c["text"]] for c in candidates]
    # Batch rerank; reranker supports batch but we'll do sequential for clarity, or use predict in batches of 64
    batch_size = 64
    all_scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        scores = reranker.predict(batch)
        if isinstance(scores, float):
            scores = [scores]
        all_scores.extend(scores)

    # Normalize reranker scores to 0-1 (min-max)
    scores_arr = np.array(all_scores)
    min_s = scores_arr.min()
    max_s = scores_arr.max()
    norm = (scores_arr - min_s) / (max_s - min_s + 1e-9)

    # Blend: 0.7 * hybrid_score_norm + 0.3 * norm_reranker
    # Assume candidates already have a "score" field (e.g., from RRF) that is normalized 0-1.
    for i, c in enumerate(candidates):
        hybrid_score_norm = c.get("score", 0)  # should already be normalized
        reranker_norm = norm[i]
        final = 0.7 * hybrid_score_norm + 0.3 * reranker_norm
        c["reranker_score"] = float(reranker_norm)
        c["final_score"] = final

    # Sort descending by final_score, keep top_k
    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return candidates[:top_k]