from __future__ import annotations
from typing import Sequence
import os
import numpy as np
import truststore

API_AVAILABLE = True
try:
    from goldmansachs.openai import OpenAI  # type: ignore
except Exception:
    API_AVAILABLE = False

truststore.inject_into_ssl()

DEFAULT_API_EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-en-icl")
DEFAULT_SBERT_MODEL = os.environ.get("SBERT_MODEL", "all-MiniLM-L6-v2")

class EmbeddingService:
    """API-first embedding client with Sentence-Transformers fallback."""
    def __init__(self, api_model: str | None = None, sbert_model: str | None = None):
        self.api_model = api_model or DEFAULT_API_EMBED_MODEL
        self.sbert_model_name = sbert_model or DEFAULT_SBERT_MODEL
        self.use_local = os.environ.get("USE_LOCAL_EMBEDS", "false").lower() in ("1","true","yes")
        self.api_client = None
        if API_AVAILABLE and not self.use_local:
            try:
                self.api_client = OpenAI()  # type: ignore
            except Exception:
                self.api_client = None
        self._sbert = None

    def _embed_api(self, texts: Sequence[str]) -> np.ndarray:
        if not self.api_client:
            raise RuntimeError("API client unavailable")
        out = self.api_client.embeddings.create(model=self.api_model, input=list(texts))  # type: ignore
        vecs = [row.embedding for row in out.data]  # type: ignore[attr-defined]
        return np.array(vecs, dtype="float32")

    def _load_sbert(self):
        if self._sbert is None:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(self.sbert_model_name)

    def _embed_local(self, texts: Sequence[str]) -> np.ndarray:
        self._load_sbert()
        vecs = self._sbert.encode(list(texts), normalize_embeddings=True)
        return np.array(vecs, dtype="float32")

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if self.use_local:
            return self._embed_local(texts)
        try:
            return self._embed_api(texts)
        except Exception:
            return self._embed_local(texts)