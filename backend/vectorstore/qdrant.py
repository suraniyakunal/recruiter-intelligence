import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams
from typing import List, Dict, Optional

COLLECTION_NAME = "candidates"

client = QdrantClient(location=":memory:")   # embedded, ephemeral

def create_collection():
    """Create or recreate the collection with dense + sparse vector config."""
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=384, distance=Distance.COSINE),  # bge-small-en-v1.5 dim
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)  # in-memory for speed
            )
        }
    )

def upsert_candidate(candidate_id: int, dense_vec: List[float], sparse_vec: dict, payload: dict):
    """
    Upsert a point into Qdrant.
    sparse_vec: {'indices': [...], 'values': [...]}
    """
    point = rest.PointStruct(
        id=candidate_id,  # use same integer as DB primary key
        vector={
            "dense": dense_vec,
            "sparse": sparse_vec
        },
        payload=payload
    )
    client.upsert(collection_name=COLLECTION_NAME, points=[point])

def hybrid_search(
    query_dense: List[float],
    query_sparse: dict,
    top_k: int = 100,
    filters: Optional[dict] = None
) -> List[rest.ScoredPoint]:
    """
    Perform hybrid search using RRF.
    filters can be Qdrant filter syntax, e.g. {"must": [...]}
    """
    prefetch = [
        rest.Prefetch(
            query=query_dense,
            using="dense",
            limit=top_k * 2,
        ),
        rest.Prefetch(
            query=rest.SparseVector(**query_sparse),
            using="sparse",
            limit=top_k * 2,
        ),
    ]

    query_filter = None
    if filters:
        # Convert our simple filter dict to Qdrant filter; implement as needed
        pass

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=rest.FusionQuery(fusion=rest.Fusion.RRF),
        limit=top_k,
        filter=query_filter,
        with_payload=True,
    )
    return results.points