from __future__ import annotations
from typing import Sequence, Optional
import numpy as np
import truststore
from openai import OpenAI

truststore.inject_into_ssl()

DEFAULT_EMBED_MODEL = "BAAI/bge-en-icl"

class EmbeddingService:
    def __init__(self, model: str = DEFAULT_EMBED_MODEL, client: Optional[OpenAI] = None):
        self.model = model
        self.client = client or OpenAI()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = self.client.embeddings.create(model=self.model, input=list(texts))
        vecs = [row.embedding for row in out.data]
        return np.array(vecs, dtype="float32")
