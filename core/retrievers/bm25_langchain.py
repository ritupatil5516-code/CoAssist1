from __future__ import annotations
from typing import List
from dataclasses import dataclass
from loguru import logger
from llama_index.core.schema import TextNode, NodeWithScore

# Prefer LangChain when available, fallback to local rank-bm25
try:
    from langchain_community.retrievers import BM25Retriever as LCBM25
    from langchain_core.documents import Document
    HAS_LANGCHAIN = True
    LC_PATH = "langchain_community.retrievers.BM25Retriever"
except Exception:
    try:
        from langchain.retrievers import BM25Retriever as LCBM25
        from langchain.schema import Document
        HAS_LANGCHAIN = True
        LC_PATH = "langchain.retrievers.BM25Retriever"
    except Exception:
        HAS_LANGCHAIN = False
        LC_PATH = None

from rank_bm25 import BM25Okapi

@dataclass
class _LocalBM25:
    nodes: List[TextNode]
    top_k: int

    def __post_init__(self):
        texts = [n.get_content() for n in self.nodes]
        self._tokens = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(self._tokens)
        logger.info("BM25 backend: Local rank-bm25 (fallback)")

    def retrieve(self, query: str) -> List[NodeWithScore]:
        qtok = query.lower().split()
        scores = self._bm25.get_scores(qtok)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.top_k]
        return [NodeWithScore(node=self.nodes[i], score=float(scores[i])) for i in idxs]


class LangChainBM25Retriever:
    def __init__(self, nodes: List[TextNode], top_k: int = 50):
        self.nodes = nodes
        self.top_k = top_k

        if HAS_LANGCHAIN:
            docs = [Document(page_content=n.get_content(), metadata=dict(n.metadata or {})) for n in nodes]
            try:
                self._lc = LCBM25.from_documents(docs)
                logger.info(f"BM25 backend: {LC_PATH}")
            except Exception as e:
                logger.warning(f"Failed to init LangChain BM25 ({e}); using local fallback.")
                self._lc = None
                self._local = _LocalBM25(nodes, top_k)
        else:
            self._lc = None
            self._local = _LocalBM25(nodes, top_k)

    @classmethod
    def from_nodes(cls, nodes: List[TextNode], similarity_top_k: int = 50):
        return cls(nodes, top_k=similarity_top_k)

    def retrieve(self, query: str) -> List[NodeWithScore]:
        if self._lc is None:
            return self._local.retrieve(query)

        # Try to use scores from LC retriever if exposed
        try:
            scores = self._lc.bm25.get_scores(query.lower().split())
            idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.top_k]
            return [NodeWithScore(node=self.nodes[i], score=float(scores[i])) for i in idxs]
        except Exception:
            # Fallback: rely on ordered docs
            docs = self._lc.get_relevant_documents(query)[: self.top_k]
            content2idx = {self.nodes[i].get_content(): i for i in range(len(self.nodes))}
            out: List[NodeWithScore] = []
            base = 1.0
            step = 1.0 / max(1, len(docs))
            for rank, d in enumerate(docs):
                i = content2idx.get(d.page_content, None)
                if i is None: 
                    continue
                out.append(NodeWithScore(node=self.nodes[i], score=base - rank * step))
            return out
