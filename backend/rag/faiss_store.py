from __future__ import annotations
from typing import List, Tuple
import numpy as np, faiss
from backend.embeddings.client import EmbeddingService
from backend.rag.types import Chunk

class FAISSStore:
    def __init__(self, chunks: List[Chunk], embedder: EmbeddingService | None = None):
        self.chunks = chunks
        self.embedder = embedder or EmbeddingService()
        texts = [c.text for c in chunks]
        embs = self.embedder.embed(texts)  # (N, D)
        faiss.normalize_L2(embs)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)

    def search(self, query: str, k: int = 20) -> List[Tuple[Chunk, float]]:
        q = self.embedder.embed([query])
        faiss.normalize_L2(q)
        D, I = self.index.search(q, k)
        out: List[Tuple[Chunk, float]] = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx == -1:
                continue
            out.append((self.chunks[idx], float(score)))
        return out
