"""
SecureRAG — File Loader
========================
Handles extraction of text from:
  • .txt  files
  • .pdf  files  (via PyPDF2)
  • .docx files  (via python-docx)
  • .png / .jpg / .jpeg / .webp images (via OpenAI Vision)
"""

from __future__ import annotations
import io
import base64
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from a plain .txt file."""
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file using PyPDF2."""
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except ImportError:
        return "[ERROR: PyPDF2 not installed. Run: pip install PyPDF2]"
    except Exception as e:
        return f"[ERROR extracting PDF: {e}]"


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a .docx file using python-docx."""
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return text.strip()
    except ImportError:
        return "[ERROR: python-docx not installed. Run: pip install python-docx]"
    except Exception as e:
        return f"[ERROR extracting DOCX: {e}]"


def extract_text_from_image(file_bytes: bytes, filename: str) -> str:
    """
    Extract text and description from an image using OpenAI Vision (GPT-4o).
    Sends the image as base64 and asks the model to describe + extract any text.
    """
    try:
        # Determine media type
        ext = filename.lower().split(".")[-1]
        media_type_map = {
            "jpg" : "image/jpeg",
            "jpeg": "image/jpeg",
            "png" : "image/png",
            "webp": "image/webp",
            "gif" : "image/gif",
        }
        media_type = media_type_map.get(ext, "image/jpeg")

        # Encode to base64
        b64_image = base64.b64encode(file_bytes).decode("utf-8")

        # Call OpenAI Vision
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{b64_image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Please do two things:\n"
                                "1. Extract ALL text visible in this image (OCR).\n"
                                "2. Describe the image in detail (what it shows, charts, diagrams, etc.).\n\n"
                                "Format your response as:\n"
                                "EXTRACTED TEXT:\n[all text from the image]\n\n"
                                "IMAGE DESCRIPTION:\n[detailed description]"
                            ),
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"[ERROR extracting image content: {e}]"


def extract_text(file_bytes: bytes, filename: str) -> tuple[str, str]:
    """
    Main dispatcher — detects file type and extracts text.
    Returns (extracted_text, file_type_label).
    """
    ext = filename.lower().split(".")[-1]

    if ext == "txt":
        return extract_text_from_txt(file_bytes), "📄 Text File"

    elif ext == "pdf":
        return extract_text_from_pdf(file_bytes), "📕 PDF File"

    elif ext == "docx":
        return extract_text_from_docx(file_bytes), "📘 Word Document"

    elif ext in ("png", "jpg", "jpeg", "webp", "gif"):
        return extract_text_from_image(file_bytes, filename), "🖼️ Image (Vision OCR)"

    else:
        return f"[Unsupported file type: .{ext}]", "❓ Unknown"
