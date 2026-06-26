import json
import asyncio
from backend.pipelines.retrieval_pipeline import retrieve_candidates
from backend.evaluation.ndcg import ndcg_at_k
from backend.evaluation.recall import recall_at_k

async def run_benchmark(test_cases_path: str):
    with open(test_cases_path, "r") as f:
        test_cases = [json.loads(line) for line in f]

    ndcg_scores = []
    recall_scores = []
    k = 10  # or from config

    for case in test_cases:
        jd_data = {
            "text": case["jd_text"],
            "skills": case.get("skills", []),
            "filters": case.get("filters", {})
        }
        relevant_ids = set(case["relevant_candidate_ids"])
        retrieved = await retrieve_candidates(jd_data, top_k=k, filters=case.get("filters"))

        retrieved_ids = [c["candidate_id"] for c in retrieved]
        # Build relevance list: 1 if in relevant set else 0
        relevances = [1 if rid in relevant_ids else 0 for rid in retrieved_ids]
        ndcg = ndcg_at_k(relevances, k)
        recall = recall_at_k(set(retrieved_ids), relevant_ids, k)

        ndcg_scores.append(ndcg)
        recall_scores.append(recall)

    print(f"Average NDCG@{k}: {sum(ndcg_scores)/len(ndcg_scores):.4f}")
    print(f"Average Recall@{k}: {sum(recall_scores)/len(recall_scores):.4f}")