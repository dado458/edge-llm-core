"""
Markdown-aware chunker for product catalogs.
Splits first on headers, then on paragraphs if a section is too long.
Returns (chunk_id, chunk_text) pairs — IDs are stable hashes so re-uploading
the same content is idempotent in ChromaDB.
"""
import hashlib
import re


def chunk_markdown(text: str, max_chunk_chars: int = 800) -> list[tuple[str, str]]:
    """
    Split markdown text into (id, content) pairs suitable for embedding.

    Strategy:
    1. Split on level-1/2/3 headers — each header starts a new chunk.
    2. If a section exceeds max_chunk_chars, further split on blank lines.
    3. IDs are 8-char MD5 prefixes of the content for idempotent upserts.
    """
    # Split on header boundaries, keeping the header with its section
    sections = re.split(r"(?=\n#{1,3} |\A#{1,3} )", text)

    raw_chunks: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= max_chunk_chars:
            raw_chunks.append(section)
        else:
            # Section too long — split on paragraphs
            paragraphs = [p.strip() for p in re.split(r"\n{2,}", section) if p.strip()]
            current = ""
            for para in paragraphs:
                if not current:
                    current = para
                elif len(current) + len(para) + 2 <= max_chunk_chars:
                    current = f"{current}\n\n{para}"
                else:
                    raw_chunks.append(current)
                    current = para
            if current:
                raw_chunks.append(current)

    return [(_stable_id(c), c) for c in raw_chunks if c]


def _stable_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
