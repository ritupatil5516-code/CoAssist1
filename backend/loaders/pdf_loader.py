from __future__ import annotations

def extract_pdf_text(path: str) -> str:
    try:
        import PyPDF2
        pages = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                try:
                    pages.append(p.extract_text() or "")
                except Exception:
                    continue
        return "\n".join(pages)
    except Exception:
        return ""
