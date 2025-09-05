from __future__ import annotations
import os, truststore
truststore.inject_into_ssl()

from llama_index.llms.openai import OpenAI as LILLM
from llama_index.embeddings.openai import OpenAIEmbedding

def make_llm():
    return LILLM(model=os.getenv("CHAT_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
                 base_url=os.getenv("OPENAI_BASE_URL") or None)

def make_embed_model():
    return OpenAIEmbedding(model=os.getenv("EMBED_MODEL", "Qwen/Qwen3-8B-Embedding"),
                           base_url=os.getenv("OPENAI_BASE_URL") or None)
