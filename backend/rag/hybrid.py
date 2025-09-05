def hybrid_merge(vec, lex, alpha=0.6, k=10):
    combined = {}
    mapping = {}
    for c, s in vec:
        key = id(c); combined[key] = combined.get(key, 0.0) + alpha * s; mapping[key] = c
    for c, s in lex:
        key = id(c); combined[key] = combined.get(key, 0.0) + (1 - alpha) * s; mapping[key] = c
    ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [(mapping[i], score) for i, score in ranked]
