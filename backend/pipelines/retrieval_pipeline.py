import asyncio
from typing import List, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from backend.db.database import AsyncSessionLocal, get_db
from backend.db.models import Candidate, JobRun, CandidateScore
from backend.models.embeddings import get_model
from backend.vectorstore.qdrant import hybrid_search
from backend.keyword.bm25 import BM25Vocab  # You'll need to import BM25Builder here for loading
from backend.models.reranker import rerank_candidates
from sklearn.preprocessing import MinMaxScaler

async def retrieve_candidates(
    jd_data: dict,  # {"text": "...", "skills": [...], "filters": {...}}
    top_k: int = 50,
    filters: Optional[dict] = None
):
    """
    1. Normalize JD skills
    2. Build dense embedding for JD text + skills
    3. Build sparse query vector using saved BM25 vocab
    4. Hybrid search Qdrant
    5. Rerank top 300
    6. Save JobRun and CandidateScore
    """
    # Load BM25 vocab
    vocab = BM25Vocab.load("data/bm25_vocab.json")
    model = get_model()

    jd_text = jd_data["text"]
    jd_skills = jd_data.get("skills", [])
    # Normalize skills? Assume they're already normalized or use normalizer (optional)
    # For embedding, combine text and skills
    query_text = jd_text + " " + " ".join(jd_skills)

    # Get dense embedding
    dense = model.encode([query_text]).tolist()[0]

    # Get sparse query vector
    sparse = vocab.encode_query(query_text)  # implement encode_query in BM25Vocab (simpler BM25 query encoding)

    # Hybrid search
    raw_results = hybrid_search(
        query_dense=dense,
        query_sparse=sparse,
        top_k=300,  # fetch more for reranking
        filters=filters
    )

    # Format candidates
    candidates = []
    for point in raw_results:
        candidates.append({
            "candidate_id": point.id,  # integer ID matching DB
            "score": point.score,       # RRF score (0-1)
            "text": point.payload.get("raw_text_snippet", ""),  # or full raw_text from DB
        })

    # Normalize RRF scores to 0-1 (they already are, but just ensure)
    scaler = MinMaxScaler()
    scores = [[c["score"]] for c in candidates]
    norm_scores = scaler.fit_transform(scores).flatten()
    for i, c in enumerate(candidates):
        c["score"] = float(norm_scores[i])

    # Get full raw_text from database for reranking (we stored snippet, better to fetch from DB)
    async with AsyncSessionLocal() as session:
        ids = [c["candidate_id"] for c in candidates]
        db_candidates = await session.run_sync(lambda s: s.query(Candidate).filter(Candidate.id.in_(ids)).all())
        id_to_text = {c.id: c.raw_text for c in db_candidates}
        for c in candidates:
            c["text"] = id_to_text.get(c["candidate_id"], "")

    # Rerank
    reranked = rerank_candidates(jd_text, candidates, top_k=top_k)

    # Store results in DB
    async with AsyncSessionLocal() as session:
        async with session.begin():
            job_run = JobRun(jd_text=jd_text, filters=filters)
            session.add(job_run)
            await session.flush()

            for rank, c in enumerate(reranked, start=1):
                score_obj = CandidateScore(
                    job_run_id=job_run.id,
                    candidate_id=c["candidate_id"],
                    bm25_score=None,  # not captured separately; we can add if needed
                    vector_score=None,
                    reranker_score=c.get("reranker_score"),
                    final_score=c["final_score"],
                    rank=rank
                )
                session.add(score_obj)
        await session.commit()
    return reranked