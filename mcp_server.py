"""
mcp_server.py — MCP (Model Context Protocol) server for MediBot.

Exposes document tools that the LLM can call on-demand:
  - search_medical_knowledge(query)   → ranked text chunks from docs
  - list_documents()                  → what's loaded
  - get_document_summary(doc_name)    → full summary of one doc

Run standalone:  python mcp_server.py
Or imported by server.py as a local tool registry.
"""

from __future__ import annotations
import os
import re
import json
import logging
from pathlib import Path
from typing import Any

import pdfplumber
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal document store (same chunking logic, but now tool-driven)
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"\s{3,}", "\n\n", text)
    return text.strip()


def _split_into_chunks(text: str, chunk_size: int = 400, overlap: int = 80):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
        i += chunk_size - overlap
    return chunks


def _load_pdf(path: str) -> str:
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    return _clean_text("\n\n".join(parts))


def _load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return _clean_text(f.read())


def _load_document(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return _load_pdf(path)
    elif ext in (".txt", ".md"):
        return _load_txt(path)
    raise ValueError(f"Unsupported file type: {ext}")


class _DocStore:
    def __init__(self):
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self.doc_texts: dict[str, str] = {}   # full text per doc label
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    def add(self, file_path: str, label: str | None = None) -> int:
        label = label or Path(file_path).name
        raw = _load_document(file_path)
        self.doc_texts[label] = raw
        new_chunks = _split_into_chunks(raw)
        self.chunks.extend(new_chunks)
        self.sources.extend([label] * len(new_chunks))
        self._rebuild()
        logger.info(f"MCP store: loaded '{label}' → {len(new_chunks)} chunks")
        return len(new_chunks)

    def _rebuild(self):
        if not self.chunks:
            return
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_features=20_000,
            sublinear_tf=True, stop_words="english",
        )
        self._matrix = self._vectorizer.fit_transform(self.chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.chunks or self._vectorizer is None:
            return []
        q_vec = self._vectorizer.transform([query])
        sims: np.ndarray = cosine_similarity(q_vec, self._matrix).flatten()
        indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in indices:
            score = float(sims[idx])
            if score < 0.01:
                continue
            results.append({
                "source": self.sources[idx],
                "score": round(score, 4),
                "text": self.chunks[idx],
            })
        return results

    def list_docs(self) -> list[str]:
        return list(self.doc_texts.keys())

    def get_full_text(self, label: str) -> str | None:
        return self.doc_texts.get(label)


# ---------------------------------------------------------------------------
# Global store — populated at import time from ./docs/
# ---------------------------------------------------------------------------

_store = _DocStore()


def _bootstrap(docs_dir: str = "docs"):
    p = Path(docs_dir)
    p.mkdir(exist_ok=True)
    readme = p / "README.txt"
    if not readme.exists():
        readme.write_text(
            "Drop your PDF or TXT medical reference documents here.\n"
            "MediBot MCP server will automatically expose them as tools.\n"
        )
    supported = list(p.glob("*.pdf")) + list(p.glob("*.txt")) + list(p.glob("*.md"))
    for f in supported:
        if f.name == "README.txt":
            continue
        try:
            _store.add(str(f), label=f.name)
        except Exception as e:
            logger.warning(f"Could not load {f.name}: {e}")


_bootstrap()


# ---------------------------------------------------------------------------
# MCP Tool Definitions  (JSON-Schema style, ready for Groq tool_choice)
# ---------------------------------------------------------------------------

MCP_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_medical_knowledge",
            "description": (
                "Search the loaded medical reference documents for information relevant "
                "to a clinical question or symptom. Returns ranked text excerpts with source labels."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Clinical query or symptom description to search for.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 4, max 8).",
                        "default": 4,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List all medical reference documents currently loaded in the knowledge base.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_summary",
            "description": "Get the full text of a specific loaded medical document by its name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_name": {
                        "type": "string",
                        "description": "Exact document name as returned by list_documents.",
                    }
                },
                "required": ["document_name"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatcher — called by server.py when the LLM requests a tool
# ---------------------------------------------------------------------------

def dispatch_tool(name: str, arguments: dict) -> str:
    """Execute a tool call and return a JSON string result."""
    try:
        if name == "search_medical_knowledge":
            query = arguments["query"]
            top_k = min(int(arguments.get("top_k", 4)), 8)
            results = _store.search(query, top_k=top_k)
            if not results:
                return json.dumps({"results": [], "message": "No relevant documents found."})
            return json.dumps({"results": results})

        elif name == "list_documents":
            docs = _store.list_docs()
            return json.dumps({
                "documents": docs,
                "total_chunks": len(_store.chunks),
                "message": f"{len(docs)} document(s) loaded." if docs else "No documents loaded yet.",
            })

        elif name == "get_document_summary":
            doc_name = arguments["document_name"]
            text = _store.get_full_text(doc_name)
            if text is None:
                return json.dumps({"error": f"Document '{doc_name}' not found."})
            # Return first 2000 chars as a summary
            summary = text[:2000] + ("..." if len(text) > 2000 else "")
            return json.dumps({"document": doc_name, "summary": summary, "total_chars": len(text)})

        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    except Exception as e:
        logger.error(f"Tool '{name}' failed: {e}")
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Public helpers used by server.py
# ---------------------------------------------------------------------------

def add_document(file_path: str, label: str | None = None) -> dict:
    n = _store.add(file_path, label)
    return {"chunks_added": n, "total_chunks": len(_store.chunks), "documents": _store.list_docs()}


def store_summary() -> dict:
    return {"total_chunks": len(_store.chunks), "documents": _store.list_docs()}
