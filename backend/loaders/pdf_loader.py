import PyPDF2

def extract_pdf_text(path: str) -> str:
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception:
        return ""
