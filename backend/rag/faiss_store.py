import faiss, numpy as np
from backend.embeddings.client import EmbeddingService
from backend.rag.types import Chunk

class FAISSStore:
    def __init__(self, chunks, embedder=None):
        self.chunks = chunks
        self.embedder = embedder or EmbeddingService()
        texts = [c.text for c in chunks]
        embs = self.embedder.embed(texts)
        faiss.normalize_L2(embs)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)
    def search(self, q: str, k: int = 20):
        qv = self.embedder.embed([q])
        faiss.normalize_L2(qv)
        D, I = self.index.search(qv, k)
        return [(self.chunks[idx], float(score)) for score, idx in zip(D[0], I[0]) if idx != -1]
