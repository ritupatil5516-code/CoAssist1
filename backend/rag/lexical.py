from rank_bm25 import BM25Okapi
from backend.rag.types import Chunk

class BM25Store:
    def __init__(self, chunks):
        self.chunks = chunks
        self.tokens = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(self.tokens)
    def search(self, q: str, k: int = 20):
        qtok = q.lower().split()
        scores = self.bm25.get_scores(qtok)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in idxs]
