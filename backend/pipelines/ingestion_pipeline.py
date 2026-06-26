import json
import asyncio
from pathlib import Path
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from collections import defaultdict
import math

from db.database import AsyncSessionLocal, init_db
from db.models import Candidate
from skills.normalizer import normalize_skills
from models.embeddings import get_model
from vectorstore.qdrant import create_collection, upsert_candidate
from skills.bm25 import BM25Vocab

# A minimal BM25 implementation

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
    bm25 = BM25Vocab()
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