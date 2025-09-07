# core/config.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import os
import yaml

_DEFAULTS: Dict[str, Any] = {
    "llm": {"style_profile": "concise"},
    "retrieval": {
        "use_hybrid": True,
        "alpha": 0.60,
        "candidates_n": 40,
        "final_k": 8,
        "freshness_lambda_per_day": 0.01,
    },
    "reranker": {"name": "llm"},
    "ui": {"show_retrieved_context": True},
}

def _deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out

def load_config(path: str | Path = "config/app.yaml") -> Dict[str, Any]:
    cfg = dict(_DEFAULTS)
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
        cfg = _deep_update(cfg, file_cfg)

    # Optional env overrides (useful in CI/containers)
    cfg["retrieval"]["use_hybrid"] = _to_bool(
        os.getenv("USE_HYBRID", str(cfg["retrieval"]["use_hybrid"]))
    )
    cfg["retrieval"]["alpha"] = float(os.getenv("HYBRID_ALPHA", cfg["retrieval"]["alpha"]))
    cfg["retrieval"]["candidates_n"] = int(os.getenv("CANDIDATES_N", cfg["retrieval"]["candidates_n"]))
    cfg["retrieval"]["final_k"] = int(os.getenv("FINAL_K", cfg["retrieval"]["final_k"]))
    cfg["retrieval"]["freshness_lambda_per_day"] = float(
        os.getenv("FRESHNESS_LAMBDA_PER_DAY", cfg["retrieval"]["freshness_lambda_per_day"])
    )
    cfg["reranker"]["name"] = os.getenv("RERANKER", cfg["reranker"]["name"])
    cfg["llm"]["style_profile"] = os.getenv("STYLE_PROFILE", cfg["llm"]["style_profile"])
    cfg["ui"]["show_retrieved_context"] = _to_bool(
        os.getenv("SHOW_RETRIEVED_CONTEXT", str(cfg["ui"]["show_retrieved_context"]))
    )
    return cfg

def _to_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}