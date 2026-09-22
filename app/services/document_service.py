from pathlib import Path

import pymupdf
from fastapi import UploadFile


ALLOWED_EXTENSIONS = {".pdf", ".txt"}


async def extract_text(file: UploadFile) -> str:
    """Extract text from a PDF or TXT file."""

    if not file.filename:
        raise ValueError("Filename is missing.")

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Only PDF and TXT files are supported.")

    file_content = await file.read()

    if not file_content:
        raise ValueError("Uploaded file is empty.")

    if extension == ".txt":
        return file_content.decode("utf-8")

    if extension == ".pdf":
        pdf = pymupdf.open(stream=file_content, filetype="pdf")

        try:
            text = "\n".join(page.get_text() for page in pdf)
        finally:
            pdf.close()

        return text

    # This should never be reached because of the extension check.
    raise ValueError("Unsupported file type.")