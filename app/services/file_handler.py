from PyPDF2 import PdfReader
from docx import Document
from fastapi import UploadFile
import io


async def extract_text_from_file(file: UploadFile) -> str:
    filename = file.filename.lower()
    content = await file.read()

    if filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    elif filename.endswith(".docx"):
        doc = Document(io.BytesIO(content))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text

    elif filename.endswith(".txt"):
        return content.decode("utf-8", errors="ignore")

    else:
        raise ValueError("Unsupported file type. Allowed: PDF, DOCX, TXT.")
