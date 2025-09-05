from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Chunk:
    text: str
    source: str
    meta: Dict[str, Any]
