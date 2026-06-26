from backend.skills.normalizer import normalize_skills
# Placeholder: we'll build during ingestion pipeline
class BM25Vocab:
    def __init__(self):
        self.idf = {}
        self.avgdl = 0
        self.doc_count = 0
        self.token2id = {}
        self.id2token = {}

    def fit(self, documents: List[str]):
        # tokenization, idf calculation
        pass

    def encode_query(self, text: str) -> dict:
    # """BM25 query encoding: weights only IDF part (no length normalization for query)."""
    tokens = self.tokenize(text)
    tf = defaultdict(int)
    for t in tokens:
        tf[t] += 1
    indices, values = [], []
    for t, freq in tf.items():
        if t in self.vocab:
            idx = self.vocab[t]
            idf_val = self.idf[t]
            # For query, BM25 often uses (k1+1)*freq/(freq+k1) * idf, but we can use simpler: idf * freq
            # We'll use idf * ( (k1+1)*freq/(freq+k1) )
            k1 = 1.2
            weight = idf_val * ((k1 + 1) * freq) / (freq + k1)
            indices.append(idx)
            values.append(weight)
    return {"indices": indices, "values": values}