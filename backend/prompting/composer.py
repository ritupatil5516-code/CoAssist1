from __future__ import annotations
from typing import List, Tuple
from textwrap import dedent
from backend.rag.types import Chunk
from .loader import PromptPack

def make_system(pack: PromptPack) -> str:
    return "\n\n".join([
        pack.system.strip(),
        "----",
        pack.retrieval.strip(),
        "----",
        pack.answer_style.strip(),
        "----",
        "Glossary (for disambiguation; cite if used):\n" + pack.glossary.strip(),
    ])

def make_user(pack: PromptPack, conversation_tail: str, numbered_context: List[Tuple[Chunk, float]], question: str) -> str:
    # number the chunks into a readable block
    rows = []
    for i, (chunk, score) in enumerate(numbered_context, 1):
        rows.append(f"[{i}] source={chunk.source} meta={chunk.meta} score={score:.3f}\n{chunk.text}")
    context_block = "\n\n".join(rows)

    return dedent(f"""
    Conversation (recent):
    {conversation_tail}

    Numbered Context:
    {context_block}

    Question: {question}

    Please answer concisely, grounded only in the context. Use the citation style defined in instructions.yaml.
    If information is insufficient, follow the refusal playbook:
    ---
    {pack.refusal.strip()}
    """).strip()