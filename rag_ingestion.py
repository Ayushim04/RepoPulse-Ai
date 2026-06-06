"""
Lightweight RAG pipeline for the RepoPulse Doc-Bot.

Ingest  : scan_repository() walks the repo, chunks files into overlapping
          text blocks, and stores them in an in-memory list + inverted index.
Retrieve: find_relevant_chunks() scores candidates with BM25-lite — no
          embeddings, no external vector DB required.
Generate: AIEngine.answer_question() receives the top-k chunks as a context
          string and asks Gemini to answer grounded in the actual codebase.

Upgrade path: swap find_relevant_chunks() for sentence-transformer embeddings
+ cosine similarity without touching any calling code.
"""

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".py", ".md", ".txt", ".js", ".ts", ".jsx", ".tsx", ".rst", ".yaml", ".yml", ".toml"}
)

IGNORED_DIRS: frozenset[str] = frozenset(
    {".git", "__pycache__", "node_modules", ".venv", "venv", "env",
     ".mypy_cache", ".pytest_cache", "dist", "build", ".next"}
)

CHUNK_SIZE: int = 1_500      # target chars per chunk
CHUNK_OVERLAP: int = 200     # chars carried into the next chunk to preserve context
MAX_RETRIEVAL_CHUNKS: int = 5


@dataclass
class TextChunk:
    source_file: str
    chunk_index: int
    text: str
    tokens: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = _tokenize(self.text)


@dataclass
class RetrievalResult:
    chunk: TextChunk
    score: float


_chunk_store: list[TextChunk] = []
_inverted_index: dict[str, list[int]] = defaultdict(list)


def scan_repository(root_path: str | Path) -> int:
    """
    Walk the repository, chunk all supported text files, and rebuild the index.
    Safe to call multiple times — the index is replaced atomically on each run.
    """
    global _chunk_store, _inverted_index

    root = Path(root_path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    logger.info(f"Starting repository scan: {root}")

    new_chunks: list[TextChunk] = []
    files_read = files_skipped = 0

    for file_path in _walk_repo(root):
        try:
            text = _read_file_safe(file_path)
            if not text:
                files_skipped += 1
                continue

            relative_path = str(file_path.relative_to(root))
            chunks = _chunk_text(text=text, source_file=relative_path)
            new_chunks.extend(chunks)
            files_read += 1
            logger.debug(f"Ingested '{relative_path}' -> {len(chunks)} chunk(s)")

        except Exception as exc:
            logger.warning(f"Skipping '{file_path}': {exc}")
            files_skipped += 1

    _chunk_store = new_chunks
    _inverted_index = _build_inverted_index(new_chunks)

    logger.info(
        f"Scan complete: {files_read} files read, {files_skipped} skipped, "
        f"{len(_chunk_store)} chunks indexed."
    )
    return len(_chunk_store)


def find_relevant_chunks(query: str, top_k: int = MAX_RETRIEVAL_CHUNKS) -> list[RetrievalResult]:
    """
    Return the top-k most relevant chunks for a query using BM25-lite scoring.

    Candidate pre-filtering via the inverted index avoids scoring every chunk
    on every query, keeping retrieval fast even as the index grows.
    """
    if not _chunk_store:
        logger.warning("Chunk store is empty — call scan_repository() first.")
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    candidate_indices: set[int] = set()
    for token in query_tokens:
        candidate_indices.update(_inverted_index.get(token, []))

    if not candidate_indices:
        candidate_indices = set(range(len(_chunk_store)))

    N = len(_chunk_store)
    results: list[RetrievalResult] = []
    for idx in candidate_indices:
        chunk = _chunk_store[idx]
        score = _bm25_score(query_tokens, chunk.tokens, chunk.text, N, _inverted_index)
        if score > 0:
            results.append(RetrievalResult(chunk=chunk, score=score))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def build_context_string(results: list[RetrievalResult]) -> str:
    """Format retrieved chunks into a single context block for the AI prompt."""
    if not results:
        return "No relevant context found in the codebase."

    sections = []
    for i, r in enumerate(results, start=1):
        sections.append(
            f"--- Context {i} | Source: {r.chunk.source_file} "
            f"(chunk #{r.chunk.chunk_index}) | Score: {r.score:.3f} ---\n"
            f"{r.chunk.text}\n"
        )
    return "\n".join(sections)


def get_index_stats() -> dict:
    return {
        "total_chunks": len(_chunk_store),
        "unique_tokens": len(_inverted_index),
        "files_indexed": len({c.source_file for c in _chunk_store}),
        "index_ready": len(_chunk_store) > 0,
    }


# ---------------------------------------------------------------------------
# File walking & reading
# ---------------------------------------------------------------------------

def _walk_repo(root: Path):
    for item in root.rglob("*"):
        if any(part in IGNORED_DIRS for part in item.parts):
            continue
        if not item.is_file():
            continue
        if item.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        yield item


def _read_file_safe(file_path: Path, max_bytes: int = 500_000) -> str:
    file_size = file_path.stat().st_size
    if file_size > max_bytes:
        logger.debug(f"Skipping large file '{file_path.name}' ({file_size / 1024:.1f} KB)")
        return ""
    if file_size == 0:
        return ""

    for encoding in ("utf-8", "latin-1"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return ""


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, source_file: str) -> list[TextChunk]:
    """
    Paragraph-aware chunking with overlap.

    Splits on double newlines first, accumulates paragraphs up to CHUNK_SIZE,
    then carries CHUNK_OVERLAP chars into the next chunk. Avoids mid-sentence
    splits that degrade retrieval quality.
    """
    chunks: list[TextChunk] = []
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        return chunks

    parts: list[str] = []
    length = 0
    index = 0

    for para in paragraphs:
        para_len = len(para)

        if para_len > CHUNK_SIZE:
            if parts:
                chunks.append(_make_chunk(parts, source_file, index))
                index += 1
                parts, length = [], 0

            for line in para.split("\n"):
                if not line.strip():
                    continue
                if length + len(line) > CHUNK_SIZE and parts:
                    chunks.append(_make_chunk(parts, source_file, index))
                    index += 1
                    overlap = " ".join(parts)[-CHUNK_OVERLAP:]
                    parts = [overlap] if overlap else []
                    length = len(overlap)
                parts.append(line)
                length += len(line)
            continue

        if length + para_len > CHUNK_SIZE and parts:
            chunks.append(_make_chunk(parts, source_file, index))
            index += 1
            overlap = " ".join(parts)[-CHUNK_OVERLAP:]
            parts = [overlap] if overlap else []
            length = len(overlap)

        parts.append(para)
        length += para_len

    if parts:
        chunks.append(_make_chunk(parts, source_file, index))

    return chunks


def _make_chunk(parts: list[str], source_file: str, index: int) -> TextChunk:
    return TextChunk(source_file=source_file, chunk_index=index, text="\n\n".join(parts))


# ---------------------------------------------------------------------------
# Retrieval & scoring
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    _STOPWORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "and", "or", "but",
        "not", "this", "that", "it", "its", "i", "we", "you", "he", "she",
        "they", "my", "your", "our", "their", "what", "which", "who", "how",
        "when", "where",
    })
    tokens = re.findall(r"[a-z_][a-z0-9_]{2,}", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def _build_inverted_index(chunks: list[TextChunk]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = defaultdict(list)
    for i, chunk in enumerate(chunks):
        for token in chunk.tokens:
            index[token].append(i)
    return index


def _bm25_score(
    query_tokens: set[str],
    doc_tokens: set[str],
    doc_text: str,
    corpus_size: int,
    inverted_index: dict[str, list[int]],
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """
    BM25-lite scoring. Penalises very long documents relative to the corpus
    average, which prevents large files from dominating retrieval results.
    """
    if not query_tokens or not doc_tokens:
        return 0.0

    avg_doc_len = 300
    doc_words = re.findall(r"\b\w+\b", doc_text.lower())
    doc_len = len(doc_words)
    word_freq: dict[str, int] = defaultdict(int)
    for w in doc_words:
        word_freq[w] += 1

    score = 0.0
    for token in query_tokens:
        if token not in doc_tokens:
            continue
        tf = word_freq.get(token, 0)
        n_docs = len(inverted_index.get(token, []))
        idf = math.log((corpus_size - n_docs + 0.5) / (n_docs + 0.5) + 1)
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))

    return score
