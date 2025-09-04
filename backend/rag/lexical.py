from __future__ import annotations
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from backend.rag.types import Chunk

def _tokenize(s: str) -> List[str]:
    return s.lower().split()

class BM25Store:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.corpus_tokens = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, k: int = 20) -> List[Tuple[Chunk, float]]:
        qtok = _tokenize(query)
        scores = self.bm25.get_scores(qtok)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in idxs]
