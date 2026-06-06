"""
RepoPulse AI - RAG Ingestion Engine (Phase 2)
===============================================
Implements a lightweight Retrieval-Augmented Generation (RAG) pipeline
that allows the Doc-Bot to answer questions grounded in the actual codebase.

Architecture (no vector DB required — pure Python for hackathon speed):
  ┌─────────────────────────────────────────────────────────┐
  │  INGEST                                                  │
  │  scan_repository() → read files → chunk text            │
  │                     → store chunks in memory (list)      │
  ├─────────────────────────────────────────────────────────┤
  │  RETRIEVE                                                │
  │  find_relevant_chunks(question) → TF-IDF style keyword  │
  │  scoring → return top-K chunks as context string        │
  ├─────────────────────────────────────────────────────────┤
  │  GENERATE                                                │
  │  AIEngine.answer_question(question, context)            │
  │  → Gemini builds a codebase-aware answer                │
  └─────────────────────────────────────────────────────────┘

Design notes:
  - No external vector database (Pinecone, Chroma, etc.) is required.
    Keyword-based retrieval is fast enough for a single-repo hackathon demo
    and avoids extra infrastructure setup time.
  - Phase 3 upgrade path: swap find_relevant_chunks() for a sentence-
    transformer embedding + cosine similarity search with zero API changes.
  - The entire index is held in memory. For large repos, use a persistent
    store (SQLite FTS5, Chroma, etc.) in Phase 3.
"""

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# File extensions we will read and ingest
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".py", ".md", ".txt", ".js", ".ts", ".jsx", ".tsx", ".rst", ".yaml", ".yml", ".toml"}
)

# Directories to skip entirely during scanning
IGNORED_DIRS: frozenset[str] = frozenset(
    {".git", "__pycache__", "node_modules", ".venv", "venv", "env",
     ".mypy_cache", ".pytest_cache", "dist", "build", ".next"}
)

# Target chunk size in characters. Chunks stay below this unless a single
# paragraph is longer, in which case it's kept intact (no mid-sentence splits).
CHUNK_SIZE: int = 1_500

# Overlap between consecutive chunks so context isn't lost at boundaries
CHUNK_OVERLAP: int = 200

# Maximum number of chunks returned by a single retrieval query
MAX_RETRIEVAL_CHUNKS: int = 5


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class TextChunk:
    """
    A single unit of text extracted from a source file.

    Attributes:
        source_file: Relative path of the file this chunk came from.
        chunk_index: Zero-based position of this chunk within the file.
        text:        The raw text content of the chunk.
        tokens:      Normalised word set, pre-computed for fast retrieval scoring.
    """
    source_file: str
    chunk_index: int
    text: str
    tokens: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        # Pre-compute tokens once at creation time
        if not self.tokens:
            self.tokens = _tokenize(self.text)


@dataclass
class RetrievalResult:
    """A chunk paired with its relevance score for a given query."""
    chunk: TextChunk
    score: float


# ---------------------------------------------------------------------------
# In-memory index
# ---------------------------------------------------------------------------

# The global chunk store. Populated by scan_repository().
# In Phase 3: replace with a persistent vector store.
_chunk_store: list[TextChunk] = []

# Inverted index: token → list of chunk indices that contain the token.
# Used for fast candidate pre-filtering before full scoring.
_inverted_index: dict[str, list[int]] = defaultdict(list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_repository(root_path: str | Path) -> int:
    """
    Walk a local repository directory, read supported files, chunk the text,
    and populate the in-memory index.

    This is the main ingestion entry point. Call it once at app startup
    (or via a /api/ingest endpoint trigger) to build the knowledge base.

    Args:
        root_path: Absolute or relative path to the repository root directory.

    Returns:
        The total number of text chunks ingested.

    Example:
        chunk_count = scan_repository("/home/user/my-repo")
        print(f"Ingested {chunk_count} chunks.")
    """
    global _chunk_store, _inverted_index

    root = Path(root_path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Repository path does not exist or is not a directory: {root}")

    logger.info(f"Starting repository scan at: {root}")

    new_chunks: list[TextChunk] = []
    files_read = 0
    files_skipped = 0

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

            logger.debug(
                f"Ingested '{relative_path}' → {len(chunks)} chunk(s)"
            )

        except Exception as exc:
            # Log but never crash — one bad file should not stop the whole scan
            logger.warning(f"Skipping '{file_path}': {exc}")
            files_skipped += 1

    # Rebuild the full index atomically (swap, don't mutate while reading)
    _chunk_store = new_chunks
    _inverted_index = _build_inverted_index(new_chunks)

    logger.info(
        f"Scan complete: {files_read} files read, {files_skipped} skipped, "
        f"{len(_chunk_store)} total chunks indexed."
    )
    return len(_chunk_store)


def find_relevant_chunks(
    query: str,
    top_k: int = MAX_RETRIEVAL_CHUNKS,
) -> list[RetrievalResult]:
    """
    Retrieve the most relevant text chunks for a given query using
    TF-IDF-inspired BM25-lite scoring.

    This is a pure keyword retrieval — no embeddings, no external API.
    Fast enough for real-time queries against a typical single-repo index.

    Args:
        query: The user's natural-language question.
        top_k: Maximum number of chunks to return.

    Returns:
        A list of RetrievalResult objects sorted by descending relevance score.
        Returns an empty list if the index is empty.
    """
    if not _chunk_store:
        logger.warning(
            "find_relevant_chunks called but the chunk store is empty. "
            "Call scan_repository() first."
        )
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    # --- Candidate filtering via inverted index ---
    # Only score chunks that share at least one token with the query.
    # This avoids scoring thousands of irrelevant chunks.
    candidate_indices: set[int] = set()
    for token in query_tokens:
        candidate_indices.update(_inverted_index.get(token, []))

    if not candidate_indices:
        # Fallback: score all chunks (happens when query uses uncommon words)
        logger.debug("No candidates from inverted index — scoring all chunks.")
        candidate_indices = set(range(len(_chunk_store)))

    # --- BM25-lite scoring ---
    N = len(_chunk_store)  # total documents in corpus
    results: list[RetrievalResult] = []

    for idx in candidate_indices:
        chunk = _chunk_store[idx]
        score = _bm25_score(
            query_tokens=query_tokens,
            doc_tokens=chunk.tokens,
            doc_text=chunk.text,
            corpus_size=N,
            inverted_index=_inverted_index,
        )
        if score > 0:
            results.append(RetrievalResult(chunk=chunk, score=score))

    # Sort by score descending and return top-k
    results.sort(key=lambda r: r.score, reverse=True)
    top_results = results[:top_k]

    logger.debug(
        f"Query '{query[:60]}...' → {len(results)} candidate chunks, "
        f"returning top {len(top_results)}."
    )
    return top_results


def build_context_string(results: list[RetrievalResult]) -> str:
    """
    Concatenate retrieved chunks into a single context block for the AI prompt.

    Each chunk is prefixed with its source file path so the AI can cite it.

    Args:
        results: The list returned by find_relevant_chunks().

    Returns:
        A formatted multi-line string ready for insertion into an AI prompt.
    """
    if not results:
        return "No relevant context found in the codebase."

    sections: list[str] = []
    for i, result in enumerate(results, start=1):
        sections.append(
            f"--- Context {i} | Source: {result.chunk.source_file} "
            f"(chunk #{result.chunk.chunk_index}) | Score: {result.score:.3f} ---\n"
            f"{result.chunk.text}\n"
        )
    return "\n".join(sections)


def get_index_stats() -> dict:
    """
    Return a summary of the current in-memory index state.
    Useful for the /api/ingest/status endpoint and the dashboard widget.
    """
    return {
        "total_chunks": len(_chunk_store),
        "unique_tokens": len(_inverted_index),
        "files_indexed": len({c.source_file for c in _chunk_store}),
        "index_ready": len(_chunk_store) > 0,
    }


# ---------------------------------------------------------------------------
# Private — File Walking & Reading
# ---------------------------------------------------------------------------

def _walk_repo(root: Path):
    """
    Yield Path objects for every supported file under root,
    skipping ignored directories and non-text files.
    """
    for item in root.rglob("*"):
        # Skip ignored directories (check all parts of the path)
        if any(part in IGNORED_DIRS for part in item.parts):
            continue

        # Only process files (not symlinks to dirs, etc.)
        if not item.is_file():
            continue

        # Only process supported extensions
        if item.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        yield item


def _read_file_safe(file_path: Path, max_bytes: int = 500_000) -> str:
    """
    Read a text file safely, handling encoding errors gracefully.

    Args:
        file_path: Path to the file.
        max_bytes: Skip files larger than this to avoid memory issues.

    Returns:
        File content as a string, or empty string if unreadable/too large.
    """
    # Skip very large files (e.g. auto-generated lock files, minified JS)
    file_size = file_path.stat().st_size
    if file_size > max_bytes:
        logger.debug(
            f"Skipping large file '{file_path.name}' "
            f"({file_size / 1024:.1f} KB > {max_bytes / 1024:.0f} KB limit)"
        )
        return ""

    if file_size == 0:
        return ""

    # Try UTF-8 first, then fall back to latin-1 (never raises on any byte)
    for encoding in ("utf-8", "latin-1"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    return ""  # Give up — likely a binary file with an unexpected extension


# ---------------------------------------------------------------------------
# Private — Text Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, source_file: str) -> list[TextChunk]:
    """
    Split a document into overlapping chunks using a paragraph-aware strategy.

    Strategy:
      1. Split on double newlines (paragraph boundaries) first.
      2. Accumulate paragraphs into a chunk until CHUNK_SIZE is reached.
      3. When a chunk is finalised, carry the last CHUNK_OVERLAP chars
         into the next chunk to preserve cross-boundary context.

    This is preferable to naive character slicing because it avoids cutting
    mid-sentence, which confuses the language model.

    Args:
        text:        The full text content of a file.
        source_file: The relative file path (for metadata only).

    Returns:
        A list of TextChunk objects.
    """
    chunks: list[TextChunk] = []

    # Split into paragraphs (2+ newlines), preserving non-empty blocks
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    if not paragraphs:
        return chunks

    current_chunk_parts: list[str] = []
    current_length: int = 0
    chunk_index: int = 0

    for para in paragraphs:
        para_len = len(para)

        # If a single paragraph exceeds CHUNK_SIZE, force-split it by lines
        if para_len > CHUNK_SIZE:
            # Flush current accumulation first
            if current_chunk_parts:
                chunks.append(_make_chunk(current_chunk_parts, source_file, chunk_index))
                chunk_index += 1
                current_chunk_parts = []
                current_length = 0

            # Hard-split the oversized paragraph by lines
            for line in para.split("\n"):
                if not line.strip():
                    continue
                if current_length + len(line) > CHUNK_SIZE and current_chunk_parts:
                    chunks.append(_make_chunk(current_chunk_parts, source_file, chunk_index))
                    chunk_index += 1
                    # Carry overlap from the end of the previous chunk
                    overlap_text = " ".join(current_chunk_parts)[-CHUNK_OVERLAP:]
                    current_chunk_parts = [overlap_text] if overlap_text else []
                    current_length = len(overlap_text)
                current_chunk_parts.append(line)
                current_length += len(line)
            continue

        # Normal case: accumulate paragraphs until we hit the size limit
        if current_length + para_len > CHUNK_SIZE and current_chunk_parts:
            chunks.append(_make_chunk(current_chunk_parts, source_file, chunk_index))
            chunk_index += 1
            # Carry the tail of the previous chunk as overlap
            overlap_text = " ".join(current_chunk_parts)[-CHUNK_OVERLAP:]
            current_chunk_parts = [overlap_text] if overlap_text else []
            current_length = len(overlap_text)

        current_chunk_parts.append(para)
        current_length += para_len

    # Flush any remaining content
    if current_chunk_parts:
        chunks.append(_make_chunk(current_chunk_parts, source_file, chunk_index))

    return chunks


def _make_chunk(parts: list[str], source_file: str, index: int) -> TextChunk:
    """Assemble chunk parts into a TextChunk."""
    return TextChunk(
        source_file=source_file,
        chunk_index=index,
        text="\n\n".join(parts),
    )


# ---------------------------------------------------------------------------
# Private — Retrieval & Scoring
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """
    Normalise text into a set of lowercase word tokens.
    Removes punctuation, numbers-only tokens, and stopwords.
    """
    _STOPWORDS = frozenset(
        {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
         "have", "has", "had", "do", "does", "did", "will", "would", "could",
         "should", "may", "might", "shall", "can", "need", "dare", "ought",
         "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
         "as", "into", "through", "and", "or", "but", "not", "this", "that",
         "it", "its", "i", "we", "you", "he", "she", "they", "my", "your",
         "our", "their", "what", "which", "who", "how", "when", "where"}
    )
    tokens = re.findall(r"[a-z_][a-z0-9_]{2,}", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def _build_inverted_index(chunks: list[TextChunk]) -> dict[str, list[int]]:
    """Build a token → [chunk_index, ...] mapping for fast lookup."""
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
    Compute a BM25-lite relevance score between a query and a document.

    BM25 is a proven bag-of-words retrieval function that outperforms
    raw TF-IDF by penalising very long documents.

    Args:
        query_tokens:   Set of normalised query tokens.
        doc_tokens:     Set of normalised document tokens.
        doc_text:       Raw document text (for term frequency calculation).
        corpus_size:    Total number of documents in the index.
        inverted_index: The global inverted index (for IDF calculation).
        k1, b:          BM25 tuning parameters (standard defaults).

    Returns:
        A non-negative float relevance score.
    """
    if not query_tokens or not doc_tokens:
        return 0.0

    # Average document length (approximate — use token count as proxy)
    avg_doc_len = 300  # reasonable approximation for code/markdown files

    doc_word_list = re.findall(r"\b\w+\b", doc_text.lower())
    doc_len = len(doc_word_list)
    word_freq: dict[str, int] = defaultdict(int)
    for word in doc_word_list:
        word_freq[word] += 1

    score = 0.0
    for token in query_tokens:
        if token not in doc_tokens:
            continue

        # Term frequency in this document
        tf = word_freq.get(token, 0)

        # Inverse document frequency (how rare is this token across all docs?)
        docs_with_token = len(inverted_index.get(token, []))
        idf = math.log((corpus_size - docs_with_token + 0.5) / (docs_with_token + 0.5) + 1)

        # BM25 term score
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
        score += idf * (numerator / denominator)

    return score
