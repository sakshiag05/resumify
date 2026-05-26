"""
SecureRAG — Vector Store (Upgraded)
======================================
Changes from v1:
  * Indirect RAG-injection defense: every retrieved chunk is scanned for
    injection patterns BEFORE being included in the context block.
    Poisoned chunks are quarantined and logged — not silently forwarded.
  * Minimum similarity threshold filtering (MIN_CHUNK_SCORE in config)
  * Re-ranking: chunks are sorted by cosine score descending
  * chunk.metadata now stores injection_quarantined flag for observability
  * save() called automatically from pipeline on document add
"""

from __future__ import annotations
import os, json, logging, unicodedata, re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

from config import (
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
    TOP_K_RETRIEVAL, MIN_CHUNK_SCORE, RAG_INJECTION_PATTERNS,
)

logger = logging.getLogger("securerag.vectorstore")


@dataclass
class Chunk:
    text     : str
    doc_name : str
    chunk_id : int
    metadata : dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    chunk : Chunk
    score : float

    def cite(self) -> str:
        return f"[Source: {self.chunk.doc_name}, chunk {self.chunk.chunk_id}]"

    @property
    def is_quarantined(self) -> bool:
        return self.chunk.metadata.get("injection_quarantined", False)


class VectorStore:
    """
    FAISS-backed vector store with:
      - Indirect injection scanning on every retrieved chunk
      - Score-threshold filtering
      - Descending score re-ranking
    """

    # Compile patterns once for efficiency
    _RAG_PATTERNS = [p.lower() for p in RAG_INJECTION_PATTERNS]

    def __init__(self):
        if not _DEPS_OK:
            raise RuntimeError("pip install faiss-cpu sentence-transformers")
        self._model  : SentenceTransformer = SentenceTransformer(EMBEDDING_MODEL)
        self._index  : Optional[faiss.IndexFlatIP] = None
        self._chunks : list[Chunk] = []
        self._dim    : int = self._model.get_sentence_embedding_dimension()

    # ── Ingestion ──────────────────────────────────────────────────────────

    def add_document(self, doc_name: str, text: str) -> int:
        chunks = self._chunk_text(text, doc_name)
        if not chunks:
            return 0
        texts      = [c.text for c in chunks]
        embeddings = self._embed(texts)
        self._ensure_index()
        self._index.add(embeddings)
        self._chunks.extend(chunks)
        logger.info(f"[VS] Indexed '{doc_name}' → {len(chunks)} chunks")
        return len(chunks)

    def add_documents(self, documents: dict[str, str]) -> dict[str, int]:
        return {name: self.add_document(name, text) for name, text in documents.items()}

    # ── Retrieval (with indirect injection defense) ────────────────────────

    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list[RetrievedChunk]:
        if self._index is None or not self._chunks:
            return []
        q_emb = self._embed([query])
        k     = min(top_k * 2, len(self._chunks))   # fetch extra, filter down
        scores, indices = self._index.search(q_emb, k)

        results: list[RetrievedChunk] = []
        quarantine_count = 0

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self._chunks[idx]
            rc    = RetrievedChunk(chunk=chunk, score=float(score))

            # ── Indirect RAG-injection defense ────────────────────────────
            # Scan the chunk text for injection patterns.  If found, flag it
            # as quarantined instead of silently forwarding to the LLM.
            # This is the critical fix: v1 only guarded the user query.
            if self._chunk_is_poisoned(chunk.text):
                chunk.metadata["injection_quarantined"] = True
                quarantine_count += 1
                logger.warning(
                    f"[VS] INDIRECT INJECTION blocked in '{chunk.doc_name}' "
                    f"chunk {chunk.chunk_id}: {chunk.text[:80]!r}"
                )
                continue  # do NOT include this chunk in results

            # ── Score threshold filter ────────────────────────────────────
            if float(score) < MIN_CHUNK_SCORE:
                continue

            results.append(rc)
            if len(results) >= top_k:
                break

        if quarantine_count:
            logger.warning(f"[VS] {quarantine_count} chunk(s) quarantined (indirect injection)")

        # Re-rank by score descending (best evidence first)
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def context_block(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> str:
        hits = self.search(query, top_k)
        if not hits:
            return ""
        lines = []
        for r in hits:
            lines.append(
                f"--- {r.cite()} (similarity: {r.score:.3f}) ---\n{r.chunk.text}"
            )
        return "\n\n".join(lines)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self._index, os.path.join(directory, "index.faiss"))
        meta = [
            {"text": c.text, "doc_name": c.doc_name,
             "chunk_id": c.chunk_id, "metadata": c.metadata}
            for c in self._chunks
        ]
        with open(os.path.join(directory, "chunks.json"), "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"[VS] Saved index to {directory!r}")

    def load(self, directory: str):
        idx_path    = os.path.join(directory, "index.faiss")
        chunks_path = os.path.join(directory, "chunks.json")
        if not os.path.exists(idx_path):
            return
        self._index = faiss.read_index(idx_path)
        with open(chunks_path) as f:
            meta = json.load(f)
        self._chunks = [
            Chunk(text=m["text"], doc_name=m["doc_name"],
                  chunk_id=m["chunk_id"], metadata=m.get("metadata", {}))
            for m in meta
        ]
        logger.info(f"[VS] Loaded {len(self._chunks)} chunks from {directory!r}")

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    @property
    def document_names(self) -> list[str]:
        return list({c.doc_name for c in self._chunks})

    # ── Private helpers ────────────────────────────────────────────────────

    def _chunk_is_poisoned(self, text: str) -> bool:
        """
        Check whether a chunk contains indirect injection patterns.
        Applies the same Unicode normalisation as the query-level guardrail
        to defeat obfuscation in uploaded documents.
        """
        normalised = unicodedata.normalize("NFKC", text).lower()
        normalised = re.sub(r"[\u00ad\u200b-\u200f\u202a-\u202e\ufeff]", "", normalised)
        return any(p in normalised for p in self._RAG_PATTERNS)

    def _chunk_text(self, text: str, doc_name: str) -> list[Chunk]:
        text = text.strip()
        if len(text) <= CHUNK_SIZE:
            return [Chunk(text=text, doc_name=doc_name, chunk_id=0)]
        chunks, start, cid = [], 0, 0
        while start < len(text):
            end = start + CHUNK_SIZE
            if end < len(text):
                para = text.rfind("\n\n", start, end)
                sent = text.rfind(". ", start, end)
                bp   = max(para, sent)
                if bp > start + CHUNK_SIZE // 2:
                    end = bp + 1
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(text=chunk_text, doc_name=doc_name, chunk_id=cid))
                cid += 1
            start = end - CHUNK_OVERLAP
        return chunks

    def _embed(self, texts: list[str]) -> np.ndarray:
        embs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embs.astype(np.float32)

    def _ensure_index(self):
        if self._index is None:
            self._index = faiss.IndexFlatIP(self._dim)
