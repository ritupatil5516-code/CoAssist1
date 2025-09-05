from textwrap import dedent

def make_system(pack):
    return "\n\n".join([
        pack.system.strip(),
        "----",
        pack.retrieval.strip(),
        "----",
        pack.answer_style.strip(),
        "----",
        "Glossary:\n"+pack.glossary.strip()
    ])

def make_user(pack, conversation_tail, numbered_context, question):
    rows = []
    for i, (chunk, score) in enumerate(numbered_context, 1):
        rows.append(f"[{i}] {chunk.source} {chunk.meta} score={score:.3f}\n{chunk.text}")
    context_block = "\n\n".join(rows)
    return dedent(f"""
    Conversation (recent):
    {conversation_tail}

    Numbered Context:
    {context_block}

    Question: {question}

    Please answer concisely, grounded only in the context. 
    If insufficient info, follow the refusal playbook:
    {pack.refusal}
    """)
