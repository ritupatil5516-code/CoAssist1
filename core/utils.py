from __future__ import annotations
import re
from datetime import datetime, timezone
from typing import Optional

def parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s: return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        try:
            return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            return None

def ym_from_dt(dt: Optional[datetime]) -> Optional[str]:
    return f"{dt.year:04d}-{dt.month:02d}" if dt else None

def detect_ym(text: str) -> Optional[str]:
    m = re.search(r"(20\d{2})-(\d{2})", text)
    if m: return f"{m.group(1)}-{m.group(2)}"
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+(20\d{2})", text, re.I)
    if m:
        mm = {"jan":"01","feb":"02","mar":"03","apr":"04","may":"05","jun":"06","jul":"07",
              "aug":"08","sep":"09","sept":"09","oct":"10","nov":"11","dec":"12"}[m.group(1)[:3].lower()]
        return f"{m.group(2)}-{mm}"
    return None
