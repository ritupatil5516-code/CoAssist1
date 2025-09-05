import PyPDF2

def extract_pdf_text(path: str) -> str:
    try:
        pages = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                pages.append(p.extract_text() or "")
        return "\n".join(pages)
    except Exception:
        return ""
