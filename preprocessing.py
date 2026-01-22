import re
from typing import List

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []

    for i in range(0, len(words), size - overlap):
        chunks.append(" ".join(words[i:i + size]))

    return chunks
