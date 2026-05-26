"""
SecureRAG — Pipeline Orchestrator (Upgraded)
=============================================
Wires together: GuardrailAgent → (if cleared) → MainAgent ← VectorStore

New in upgraded version:
  * Indirect RAG injection is defended at the VectorStore layer (not here).
  * Blocked queries are recorded in MainAgent history (mark_blocked).
  * FAISS index auto-saved after each document add.
  * query_stats dict for observability dashboard.
"""

from __future__ import annotations
import os, logging
from dataclasses import dataclass, field
from typing import Optional, Generator

from agents.guardrail_agent import GuardrailAgent, GuardrailResult
from agents.main_agent       import MainAgent
from rag.vector_store        import VectorStore

logger = logging.getLogger("securerag.pipeline")

INDEX_DIR = "securerag_index"   # persisted FAISS index location


@dataclass
class PipelineResponse:
    guardrail   : GuardrailResult
    answer      : Optional[str]
    full_output : str


class SecureRAGPipeline:
    """
    Two-agent pipeline:

      User Input
          │
          ▼
      [GUARDRAIL AGENT]  ← Layer 1 (regex + unicode normalisation)
          │               ← Layer 2 (few-shot LLM classifier, fail-CLOSED)
      ┌───┴──────────────────────────────┐
    SAFE / SANITIZED                  UNSAFE
          │                               │
          ▼                               ▼
      [VECTOR STORE]              "Unable to process"
          │  ← Indirect injection        (blocked query logged)
          │    scan on every chunk
          ▼
      [MAIN AGENT]
          │  ← Token-aware history trim
          │  ← Blocked turns excluded
          ▼
      Response to User
    """

    def __init__(self):
        self._guardrail = GuardrailAgent()
        self._vs        = VectorStore()
        self._main      = MainAgent(vector_store=self._vs)

        # Load persisted index if available
        if os.path.exists(INDEX_DIR):
            try:
                self._vs.load(INDEX_DIR)
                logger.info(f"[PIPELINE] Loaded persisted index: {self._vs.total_chunks} chunks")
            except Exception as e:
                logger.warning(f"[PIPELINE] Could not load index: {e}")

        # Session-level observability counters
        self.stats: dict = {
            "total": 0, "safe": 0, "sanitized": 0, "blocked": 0,
            "rag_quarantines": 0,
        }

    # ── Streaming API ──────────────────────────────────────────────────────

    def run_guardrail(self, user_input: str) -> GuardrailResult:
        return self._guardrail.evaluate(user_input)

    def stream_answer(
        self, clean_query: str, sanitized: bool = False
    ) -> Generator[str, None, None]:
        yield from self._main.chat_stream(query=clean_query, sanitized=sanitized)

    # ── Non-streaming API (demo.py) ────────────────────────────────────────

    def query(self, user_input: str) -> PipelineResponse:
        self.stats["total"] += 1
        gr = self._guardrail.evaluate(user_input)

        if gr.is_blocked:
            self.stats["blocked"] += 1
            self._main.mark_blocked(user_input)   # exclude from future context
            full = f"{gr.report_header}\n\nI'm unable to process that request."
            return PipelineResponse(guardrail=gr, answer=None, full_output=full)

        if gr.is_sanitized:
            self.stats["sanitized"] += 1
            note = "Note: Malicious fragment removed. Processing clean intent only.\n\n"
        else:
            self.stats["safe"] += 1
            note = ""

        answer = self._main.chat(query=gr.clean_query, sanitized=gr.is_sanitized)
        full   = f"{gr.report_header}\n\n{note}{answer}"
        return PipelineResponse(guardrail=gr, answer=answer, full_output=full)

    # ── Knowledge base ─────────────────────────────────────────────────────

    def add_document(self, doc_name: str, text: str) -> int:
        n = self._vs.add_document(doc_name, text)
        self._persist_index()
        return n

    def add_documents(self, documents: dict[str, str]) -> dict[str, int]:
        results = self._vs.add_documents(documents)
        self._persist_index()
        return results

    def _persist_index(self):
        try:
            self._vs.save(INDEX_DIR)
        except Exception as e:
            logger.warning(f"[PIPELINE] Index persist failed: {e}")

    def reset_conversation(self):
        self._main.reset()
        self.stats = {
            "total": 0, "safe": 0, "sanitized": 0, "blocked": 0,
            "rag_quarantines": 0,
        }

    @property
    def knowledge_base_stats(self) -> dict:
        return {
            "total_chunks"  : self._vs.total_chunks,
            "document_names": self._vs.document_names,
        }

    def get_retrieved_chunks(self, query: str) -> list:
        hits = self._vs.search(query)
        return [
            (r.chunk.doc_name, r.chunk.chunk_id, r.score, r.chunk.text)
            for r in hits
        ]
