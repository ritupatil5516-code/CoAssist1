from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import faiss

from backend.embeddings.client import EmbeddingService

class FAISSStore:
    def __init__(self, texts: List[str], embedder: Optional[EmbeddingService] = None):
        self.texts = texts
        self.embedder = embedder or EmbeddingService()
        embs = self.embedder.embed(texts)
        faiss.normalize_L2(embs)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        q = self.embedder.embed([query])
        faiss.normalize_L2(q)
        D, I = self.index.search(q, k)
        out: List[Tuple[str, float]] = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx == -1:
                continue
            out.append((self.texts[idx], float(score)))
        return out
