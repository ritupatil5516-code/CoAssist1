import os, numpy as np, truststore
truststore.inject_into_ssl()
try:
    from goldmansachs.openai import OpenAI
    API_AVAILABLE = True
except Exception:
    API_AVAILABLE = False

USE_TFIDF = os.environ.get("USE_TFIDF","false").lower() in ("1","true","yes")
DEFAULT_API_MODEL = os.environ.get("EMBED_MODEL","BAAI/bge-en-icl")

class EmbeddingService:
    def __init__(self, api_model=None):
        self.api_model = api_model or DEFAULT_API_MODEL
        self.use_tfidf = USE_TFIDF or not API_AVAILABLE
        self.api = OpenAI() if (API_AVAILABLE and not self.use_tfidf) else None
        self._tfidf = None

    def _embed_api(self, texts):
        out = self.api.embeddings.create(model=self.api_model, input=list(texts))
        return np.array([r.embedding for r in out.data], dtype="float32")

    def fit_corpus(self, corpus_texts):
        if not self.use_tfidf: return
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(max_features=4096, ngram_range=(1,2), lowercase=True)
        self._tfidf.fit(corpus_texts)

    def embed_corpus(self, texts):
        if self.use_tfidf:
            X = self._tfidf.transform(texts); return X.toarray().astype("float32")
        return self._embed_api(texts)

    def embed_query(self, texts):
        if self.use_tfidf:
            X = self._tfidf.transform(texts); return X.toarray().astype("float32")
        return self._embed_api(texts)
