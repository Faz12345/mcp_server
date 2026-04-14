"""
mcp_server.py — MCP server for MediBot.

Tool registry (6 tools, LLM decides which to call):

  LOCAL (uploaded docs, TF-IDF):
    - search_local_knowledge(query, top_k)
    - list_documents()
    - get_document_summary(document_name)

  EXTERNAL (live APIs, no key required):
    - search_pubmed(query, max_results)       → NIH PubMed abstracts
    - lookup_drug(drug_name)                  → OpenFDA label + RxNorm classification
    - lookup_condition(condition_name)        → OpenFDA FAERS adverse event reports

Priority (described in tool descriptions so LLM decides):
  - General clinical questions  → search_pubmed first
  - Drug / medication queries   → lookup_drug first
  - Institution-specific docs   → search_local_knowledge first
  - LLM can chain multiple tools in one reasoning turn
"""

from __future__ import annotations
import os, re, json, logging, time
from pathlib import Path
from urllib.parse import quote

import pdfplumber
import numpy as np
import urllib.request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared HTTP helper
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: int = 8) -> dict | list | None:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "MediBot/2.0 (educational; contact@example.com)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        logger.warning(f"HTTP GET failed [{url[:80]}...]: {e}")
        return None


# ---------------------------------------------------------------------------
# Local document store
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
        self.doc_texts: dict[str, str] = {}
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
        logger.info(f"DocStore: '{label}' → {len(new_chunks)} chunks")
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


_store = _DocStore()


def _bootstrap(docs_dir: str = "docs"):
    p = Path(docs_dir)
    p.mkdir(exist_ok=True)
    readme = p / "README.txt"
    if not readme.exists():
        readme.write_text(
            "Drop PDF or TXT medical reference documents here.\n"
            "MediBot indexes them alongside live PubMed / OpenFDA / RxNorm APIs.\n"
        )
    for f in list(p.glob("*.pdf")) + list(p.glob("*.txt")) + list(p.glob("*.md")):
        if f.name == "README.txt":
            continue
        try:
            _store.add(str(f), label=f.name)
        except Exception as e:
            logger.warning(f"Could not load {f.name}: {e}")


_bootstrap()


# ---------------------------------------------------------------------------
# External API implementations
# ---------------------------------------------------------------------------

PUBMED_BASE  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OPENFDA_BASE = "https://api.fda.gov"
RXNORM_BASE  = "https://rxnav.nlm.nih.gov/REST"
NCBI_EMAIL   = os.environ.get("NCBI_EMAIL", "medibot@example.com")
NCBI_TOOL    = "MediBot"


def _pubmed_search(query: str, max_results: int = 4) -> list[dict]:
    """Search PubMed → return article metadata list."""
    search_url = (
        f"{PUBMED_BASE}/esearch.fcgi?db=pubmed&retmode=json"
        f"&retmax={max_results}&term={quote(query)}"
        f"&tool={NCBI_TOOL}&email={NCBI_EMAIL}"
    )
    search_data = _http_get(search_url)
    if not search_data:
        return []

    ids = search_data.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    time.sleep(0.35)  # NCBI polite rate limit

    summary_url = (
        f"{PUBMED_BASE}/esummary.fcgi?db=pubmed&retmode=json"
        f"&id={','.join(ids)}"
        f"&tool={NCBI_TOOL}&email={NCBI_EMAIL}"
    )
    summary_data = _http_get(summary_url)
    if not summary_data:
        return []

    results = []
    for uid in summary_data.get("result", {}).get("uids", []):
        art = summary_data["result"].get(uid, {})
        results.append({
            "pmid":    uid,
            "title":   art.get("title", "No title"),
            "authors": ", ".join(a.get("name", "") for a in art.get("authors", [])[:3]),
            "journal": art.get("source", ""),
            "year":    art.get("pubdate", "")[:4],
            "url":     f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
        })
    return results


def _openfda_drug(drug_name: str) -> dict:
    """Fetch FDA drug label info from OpenFDA."""
    url = (
        f"{OPENFDA_BASE}/drug/label.json"
        f"?search=openfda.brand_name:\"{quote(drug_name)}\""
        f"+openfda.generic_name:\"{quote(drug_name)}\""
        f"&limit=1"
    )
    data = _http_get(url)
    if not data or not data.get("results"):
        data = _http_get(f"{OPENFDA_BASE}/drug/label.json?search={quote(drug_name)}&limit=1")

    if not data or not data.get("results"):
        return {"error": f"'{drug_name}' not found in OpenFDA."}

    r = data["results"][0]
    openfda = r.get("openfda", {})

    def first(field, limit=600):
        val = r.get(field, [])
        return val[0][:limit] if val else None

    return {
        "drug_name":         openfda.get("brand_name", [drug_name])[0],
        "generic_name":      openfda.get("generic_name", [""])[0],
        "manufacturer":      openfda.get("manufacturer_name", [""])[0],
        "indications":       first("indications_and_usage"),
        "contraindications": first("contraindications"),
        "warnings":          first("warnings"),
        "drug_interactions": first("drug_interactions"),
        "dosage":            first("dosage_and_administration"),
        "adverse_reactions": first("adverse_reactions"),
        "source":            "OpenFDA — FDA drug label database",
    }


def _rxnorm_lookup(drug_name: str) -> dict:
    """Normalize drug name and fetch ATC drug classes via RxNorm."""
    cui_data = _http_get(f"{RXNORM_BASE}/rxcui.json?name={quote(drug_name)}&search=1")
    if not cui_data:
        return {"error": "RxNorm unreachable."}

    rxcui = cui_data.get("idGroup", {}).get("rxnormId", [None])[0]
    if not rxcui:
        return {"error": f"'{drug_name}' not found in RxNorm."}

    props_data  = _http_get(f"{RXNORM_BASE}/rxcui/{rxcui}/properties.json")
    class_data  = _http_get(f"{RXNORM_BASE}/rxclass/class/byRxcui.json?rxcui={rxcui}&relaSource=ATC")

    props = props_data.get("properties", {}) if props_data else {}
    classes = []
    if class_data:
        for item in class_data.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", []):
            cls = item.get("rxclassMinConceptItem", {})
            if cls.get("className"):
                classes.append(cls["className"])

    return {
        "rxcui":        rxcui,
        "name":         props.get("name", drug_name),
        "synonym":      props.get("synonym", ""),
        "drug_classes": list(set(classes[:5])),
        "rxnorm_url":   f"https://mor.nlm.nih.gov/RxNav/search?searchBy=RXCUI&searchTerm={rxcui}",
        "source":       "RxNorm — NIH National Library of Medicine",
    }


def _openfda_condition(condition: str) -> list[dict]:
    """Return top drugs associated with a condition in FDA adverse event reports."""
    url = (
        f"{OPENFDA_BASE}/drug/event.json"
        f"?search=patient.reaction.reactionmeddrapt:\"{quote(condition)}\""
        f"&count=patient.drug.openfda.generic_name.exact&limit=5"
    )
    data = _http_get(url)
    if not data or not data.get("results"):
        return []
    return [
        {"drug": r.get("term", ""), "adverse_event_reports": r.get("count", 0)}
        for r in data["results"][:5]
    ]


# ---------------------------------------------------------------------------
# MCP Tool Definitions
# ---------------------------------------------------------------------------

MCP_TOOLS: list[dict] = [
    # LOCAL ──────────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "search_local_knowledge",
            "description": (
                "Search locally uploaded medical reference documents (PDFs/TXTs). "
                "Best for institution-specific protocols, personal health records, or custom "
                "guidelines uploaded by the user. If no documents are loaded or results are "
                "weak, fall back to search_pubmed for general medical knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Clinical query or symptom description."},
                    "top_k": {"type": "integer", "description": "Results to return (default 4, max 8).", "default": 4},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List all locally uploaded medical reference documents in the knowledge base.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_summary",
            "description": "Get the full text of a specific locally uploaded document by its exact name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_name": {"type": "string", "description": "Exact name as returned by list_documents."}
                },
                "required": ["document_name"],
            },
        },
    },

    # EXTERNAL ───────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "search_pubmed",
            "description": (
                "Search PubMed — NIH's database of 35 million+ peer-reviewed biomedical articles. "
                "Use for any clinical question, disease mechanism, treatment efficacy, symptoms, "
                "or evidence-based medicine. Returns article titles, authors, journals, years, "
                "and direct PubMed URLs. PREFER this over local docs for general medical questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "PubMed search query, e.g. 'chest pain differential diagnosis' or 'metformin type 2 diabetes outcomes'.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Articles to return (default 4, max 8).",
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
            "name": "lookup_drug",
            "description": (
                "Look up a drug in the FDA label database (OpenFDA) and NIH RxNorm. "
                "Returns: indications, contraindications, warnings, drug interactions, dosage, "
                "adverse reactions, and ATC drug classification. Use whenever a patient mentions "
                "a specific medication name or asks about dosing/interactions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "Brand or generic drug name, e.g. 'ibuprofen', 'Metformin', 'amoxicillin'.",
                    }
                },
                "required": ["drug_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_condition",
            "description": (
                "Search the FDA Adverse Event Reporting System (FAERS) for a medical condition "
                "or symptom. Returns which drugs are most commonly associated with that condition "
                "in real-world patient reports. Useful for differential diagnosis support or "
                "understanding drug-condition relationships. Results show report counts, not causation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "condition_name": {
                        "type": "string",
                        "description": "Medical condition or symptom in MedDRA terminology, e.g. 'hypertension', 'nausea', 'rash', 'chest pain'.",
                    }
                },
                "required": ["condition_name"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def dispatch_tool(name: str, arguments: dict) -> str:
    logger.info(f"Dispatching tool: {name}({list(arguments.keys())})")
    try:
        # LOCAL
        if name == "search_local_knowledge":
            query = arguments["query"]
            top_k = min(int(arguments.get("top_k", 4)), 8)
            results = _store.search(query, top_k=top_k)
            if not results:
                return json.dumps({
                    "results": [],
                    "message": "No relevant content in local documents. Consider using search_pubmed.",
                })
            return json.dumps({"results": results, "source": "local_documents"})

        elif name == "list_documents":
            docs = _store.list_docs()
            return json.dumps({
                "documents": docs,
                "total_chunks": len(_store.chunks),
                "message": f"{len(docs)} local document(s) loaded." if docs else "No local documents uploaded yet.",
            })

        elif name == "get_document_summary":
            doc_name = arguments["document_name"]
            text = _store.get_full_text(doc_name)
            if text is None:
                return json.dumps({"error": f"'{doc_name}' not found."})
            summary = text[:2000] + ("..." if len(text) > 2000 else "")
            return json.dumps({"document": doc_name, "summary": summary, "total_chars": len(text)})

        # EXTERNAL
        elif name == "search_pubmed":
            query = arguments["query"]
            max_results = min(int(arguments.get("max_results", 4)), 8)
            articles = _pubmed_search(query, max_results)
            if not articles:
                return json.dumps({"articles": [], "message": "No PubMed results. Try rephrasing."})
            return json.dumps({
                "articles": articles,
                "source": "PubMed — NIH National Library of Medicine",
                "count": len(articles),
            })

        elif name == "lookup_drug":
            drug_name = arguments["drug_name"]
            fda_info = _openfda_drug(drug_name)
            rxn_info = _rxnorm_lookup(drug_name)
            return json.dumps({
                "fda_label": fda_info,
                "rxnorm":    rxn_info,
                "sources":   ["OpenFDA (FDA)", "RxNorm (NIH)"],
            })

        elif name == "lookup_condition":
            condition = arguments["condition_name"]
            reports = _openfda_condition(condition)
            if not reports:
                return json.dumps({
                    "condition": condition,
                    "associated_drugs": [],
                    "message": "No FAERS adverse event data found for this condition.",
                })
            return json.dumps({
                "condition":        condition,
                "associated_drugs": reports,
                "source":           "OpenFDA FAERS — FDA Adverse Event Reporting System",
                "note":             "Counts are adverse event reports; do not imply causation.",
            })

        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    except Exception as e:
        logger.error(f"Tool '{name}' failed: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Public helpers for server.py
# ---------------------------------------------------------------------------

def add_document(file_path: str, label: str | None = None) -> dict:
    n = _store.add(file_path, label)
    return {"chunks_added": n, "total_chunks": len(_store.chunks), "documents": _store.list_docs()}


def store_summary() -> dict:
    return {
        "total_chunks":  len(_store.chunks),
        "documents":     _store.list_docs(),
        "external_apis": ["PubMed (NIH)", "OpenFDA (FDA)", "RxNorm (NIH)"],
    }
