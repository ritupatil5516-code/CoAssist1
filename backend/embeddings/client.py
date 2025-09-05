import os, numpy as np, truststore
truststore.inject_into_ssl()

try:
    from goldmansachs.openai import OpenAI
    API_AVAILABLE=True
except Exception:
    API_AVAILABLE=False

DEFAULT_API_MODEL = os.environ.get("EMBED_MODEL","BAAI/bge-en-icl")
DEFAULT_SBERT = os.environ.get("SBERT_MODEL","all-MiniLM-L6-v2")

class EmbeddingService:
    def __init__(self, api_model=None, sbert_model=None):
        self.api_model = api_model or DEFAULT_API_MODEL
        self.sbert_model = sbert_model or DEFAULT_SBERT
        self.use_local = os.environ.get("USE_LOCAL_EMBEDS","false").lower() in ("1","true","yes")
        self.api = OpenAI() if API_AVAILABLE and not self.use_local else None
        self._sbert = None

    def _embed_api(self, texts):
        out = self.api.embeddings.create(model=self.api_model, input=list(texts))
        return np.array([r.embedding for r in out.data], dtype="float32")

    def _embed_local(self, texts):
        if self._sbert is None:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(self.sbert_model)
        vecs = self._sbert.encode(list(texts), normalize_embeddings=True)
        return np.array(vecs, dtype="float32")

    def embed(self, texts):
        if self.use_local or not self.api:
            return self._embed_local(texts)
        try:
            return self._embed_api(texts)
        except Exception:
            return self._embed_local(texts)
