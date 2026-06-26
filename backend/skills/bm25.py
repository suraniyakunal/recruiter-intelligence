import math
import json
from collections import defaultdict
from typing import List, Dict

class BM25Vocab:
    """
    Lightweight BM25 vocabulary builder that can:
    - fit on a corpus of documents
    - encode documents as sparse vectors (for Qdrant)
    - encode queries as sparse vectors
    - save/load vocabulary to/from JSON
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.idf = defaultdict(float)
        self.avgdl = 0.0
        self.doc_count = 0
        self.token2id = {}
        self.id2token = {}
        self.doc_len = []          # lengths of documents used for fitting

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace/lowercase tokenizer – improve if needed."""
        return text.lower().split()

    def fit(self, documents: List[str]):
        """
        Build vocabulary and compute IDF values from a list of documents.
        """
        N = len(documents)
        doc_freq = defaultdict(int)
        token_docs = []
        for doc in documents:
            tokens = self.tokenize(doc)
            token_docs.append(tokens)
            self.doc_len.append(len(tokens))
            unique = set(tokens)
            for t in unique:
                doc_freq[t] += 1

        self.avgdl = sum(self.doc_len) / N if N else 0.0
        self.token2id = {t: idx for idx, t in enumerate(doc_freq.keys())}
        self.id2token = {idx: t for t, idx in self.token2id.items()}
        self.doc_count = N

        for t, idx in self.token2id.items():
            df = doc_freq[t]
            # IDF smoothing variant (same as used in many BM25 implementations)
            self.idf[t] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def _doc_weight(self, freq: int, doc_len: int) -> float:
        """BM25 weight for a term in a document."""
        idf_val = 1.0  # will be multiplied later
        numerator = freq * (self.k1 + 1)
        denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
        return numerator / denominator

    def encode_sparse(self, text: str) -> Dict[str, list]:
        """
        Encode a document into a sparse vector dict with 'indices' and 'values'.
        """
        tokens = self.tokenize(text)
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        doc_len = len(tokens)

        indices = []
        values = []
        for t, freq in tf.items():
            if t in self.token2id:
                idx = self.token2id[t]
                idf_val = self.idf[t]
                weight = idf_val * self._doc_weight(freq, doc_len)
                indices.append(idx)
                values.append(weight)
        return {"indices": indices, "values": values}

    def encode_query(self, text: str) -> Dict[str, list]:
        """
        Encode a query – uses IDF only with BM25‑like term frequency scaling.
        """
        tokens = self.tokenize(text)
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1

        indices = []
        values = []
        for t, freq in tf.items():
            if t in self.token2id:
                idx = self.token2id[t]
                idf_val = self.idf[t]
                # Query TF scaling (standard BM25 query formula)
                weight = idf_val * ((self.k1 + 1) * freq) / (freq + self.k1)
                indices.append(idx)
                values.append(weight)
        return {"indices": indices, "values": values}

    def save(self, path: str):
        """Persist vocabulary data to a JSON file."""
        data = {
            "idf": dict(self.idf),
            "avgdl": self.avgdl,
            "doc_count": self.doc_count,
            "token2id": self.token2id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BM25Vocab":
        """Load vocabulary from a JSON file and return a BM25Vocab instance."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = cls()
        obj.idf = defaultdict(float, data["idf"])
        obj.avgdl = data["avgdl"]
        obj.doc_count = data["doc_count"]
        obj.token2id = data["token2id"]
        obj.id2token = {v: k for k, v in obj.token2id.items()}
        return obj