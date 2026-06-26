import json
import asyncio
from pathlib import Path
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from collections import defaultdict
import math

from backend.db.database import AsyncSessionLocal, init_db
from backend.db.models import Candidate
from backend.skills.normalizer import normalize_skills
from backend.models.embeddings import get_model
from backend.vectorstore.qdrant import create_collection, upsert_candidate

# A minimal BM25 implementation
class BM25Builder:
    def __init__(self):
        self.idf = defaultdict(float)
        self.doc_len = []
        self.avgdl = 0
        self.vocab = {}
        self.vocab_size = 0

    def tokenize(self, text: str) -> list[str]:
        # simple whitespace + lowercase; you can improve
        return text.lower().split()

    def fit(self, documents: list[str]):
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
        self.avgdl = sum(self.doc_len) / N if N else 0
        self.vocab = {t: idx for idx, t in enumerate(doc_freq.keys())}
        self.vocab_size = len(self.vocab)
        for t, idx in self.vocab.items():
            df = doc_freq[t]
            self.idf[t] = math.log((N - df + 0.5) / (df + 0.5) + 1)  # smoothing

    def encode_sparse(self, text: str) -> dict:
        tokens = self.tokenize(text)
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        doc_len = len(tokens)
        indices = []
        values = []
        for t, freq in tf.items():
            if t in self.vocab:
                idx = self.vocab[t]
                # BM25 term weight: idf * ((freq * (k1+1)) / (freq + k1*(1-b+b*dl/avgdl)))
                k1, b = 1.2, 0.75
                idf_val = self.idf[t]
                numerator = freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * doc_len / self.avgdl)
                weight = idf_val * numerator / denominator
                indices.append(idx)
                values.append(weight)
        return {"indices": indices, "values": values}

    def save(self, path: str):
        data = {
            "idf": dict(self.idf),
            "avgdl": self.avgdl,
            "vocab": self.vocab,
            "doc_len_avg": self.avgdl
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str):
        obj = cls()
        with open(path, "r") as f:
            data = json.load(f)
        obj.idf = defaultdict(float, data["idf"])
        obj.avgdl = data["avgdl"]
        obj.vocab = data["vocab"]
        obj.vocab_size = len(obj.vocab)
        return obj

async def run_ingestion(data_path: str):
    await init_db()
    model = get_model()
    raw_dir = Path(data_path)
    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True)

    all_docs = []
    records = []

    # Step 1: read raw JSONL, normalize skills, build text for BM25
    with open(raw_dir / "candidates.jsonl", "r") as f:
        for line in f:
            obj = json.loads(line)
            name = obj.get("name", "Unknown")
            raw_text = obj.get("raw_text", "")
            raw_skills = obj.get("skills", [])
            normalized = normalize_skills(raw_skills)
            # For BM25, combine raw_text and skills (or just skills)
            doc_text = raw_text + " " + " ".join(normalized)
            all_docs.append(doc_text)
            records.append({
                "name": name,
                "raw_text": raw_text,
                "normalized_skills": normalized,
                "doc_text": doc_text
            })

    # Step 2: Build BM25 vocabulary on all documents
    print("Building BM25 vocabulary...")
    bm25 = BM25Builder()
    bm25.fit(all_docs)
    bm25.save("data/bm25_vocab.json")
    print("BM25 vocabulary saved.")

    # Step 3: Generate embeddings in batches of 256
    batch_size = 256
    num_records = len(records)
    embeddings = []
    for i in range(0, num_records, batch_size):
        batch = [r["doc_text"] for r in records[i:i+batch_size]]
        batch_emb = model.encode(batch, show_progress_bar=False).tolist()
        embeddings.extend(batch_emb)
        print(f"Encoded {min(i+batch_size, num_records)}/{num_records}")

    # Step 4: Create Qdrant collection
    print("Creating Qdrant collection...")
    create_collection()

    # Step 5: Insert into SQLite and Qdrant
    async with AsyncSessionLocal() as session:
        async with session.begin():
            for idx, rec in enumerate(records):
                # Insert Candidate row
                cand = Candidate(
                    name=rec["name"],
                    raw_text=rec["raw_text"],
                    normalized_skills=rec["normalized_skills"],
                    embedding_id=str(idx)  # can be same as id after flush
                )
                session.add(cand)
                await session.flush()  # get candidate.id

                # Prepare sparse vector for Qdrant
                sparse_vec = bm25.encode_sparse(rec["doc_text"])
                upsert_candidate(
                    candidate_id=cand.id,
                    dense_vec=embeddings[idx],
                    sparse_vec=sparse_vec,
                    payload={
                        "name": rec["name"],
                        "normalized_skills": rec["normalized_skills"],
                        "raw_text_snippet": rec["raw_text"][:200]
                    }
                )
        await session.commit()
    print(f"Ingestion complete. {num_records} candidates indexed.")