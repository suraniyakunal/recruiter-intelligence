def recall_at_k(retrieved_ids: set, relevant_ids: set, k: int) -> float:
    retrieved_k = set(list(retrieved_ids)[:k])
    return len(retrieved_k & relevant_ids) / len(relevant_ids) if relevant_ids else 0.0

def precision_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    retrieved_k = set(retrieved_ids[:k])
    return len(retrieved_k & relevant_ids) / k if k else 0.0