import math

def dcg_at_k(relevances: list[float], k: int) -> float:
    relevances = relevances[:k]
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))

def ndcg_at_k(true_relevances: list[float], k: int) -> float:
    dcg = dcg_at_k(true_relevances, k)
    ideal = sorted(true_relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0