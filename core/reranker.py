from __future__ import annotations
from typing import List
from llama_index.postprocessor.llm_rerank import LLMRerank
from core.llm import make_llm

def rerank_nodes(nodes, query: str, k: int):
    rr = LLMRerank(choice_batch_size=8, top_n=k, llm=make_llm())
    return rr.postprocess_nodes(nodes, query_str=query)
