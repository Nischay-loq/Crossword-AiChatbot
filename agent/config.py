"""
agent/config.py
Central configuration and authority-ranking constants.
All values can be overridden via environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
KNOWLEDGE_BASE_DIR = ROOT_DIR / os.getenv("KNOWLEDGE_BASE_DIR", "knowledge-base")
ORDERS_FILE = ROOT_DIR / os.getenv("ORDERS_FILE", "data/orders.json")
CHROMA_PERSIST_DIR = ROOT_DIR / os.getenv("CHROMA_PERSIST_DIR", ".chroma_db")

def get_gemini_api_key() -> str:
    load_dotenv(override=True)
    return os.getenv("GEMINI_API_KEY", "")

def get_gemini_model() -> str:
    load_dotenv(override=True)
    return os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# ── Embeddings — local, free (sentence-transformers) ─────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = ROOT_DIR / os.getenv("CHROMA_PERSIST_DIR", ".chroma_db")
COLLECTION_NAME = "aster_row_kb"

# ── Retrieval ────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = 6          # number of chunks to retrieve
CHUNK_SIZE = 600             # characters per chunk
CHUNK_OVERLAP = 80           # character overlap between chunks

# ── Debug ────────────────────────────────────────────────────────────────────
DEBUG = os.getenv("DEBUG", "0") == "1"

# ── Document Authority Rules ─────────────────────────────────────────────────
# Documents excluded from customer answers entirely (never retrieved for customer use)
EXCLUDED_FROM_RETRIEVAL: set[str] = {
    "draft",       # status values that disqualify a doc
}
EXCLUDED_AUDIENCE: set[str] = {
    "internal",    # audience values that disqualify a doc
}
EXCLUDED_POLICY_AUTHORITY: set[str] = {
    "none",        # policy_authority values that disqualify a doc
}

# Authority score assigned during reranking (higher = more trusted)
# Format: {(status, policy_authority, audience): score}
AUTHORITY_SCORES: dict[tuple[str, str, str], int] = {
    ("active",     "official", "customer"): 100,
    ("active",     "official", "all"):      100,
    ("active",     "official", "internal"): 0,    # internal docs get score 0 = excluded
    ("superseded", "official", "customer"): 20,   # low priority, never primary source
    ("superseded", "official", "all"):      20,
    ("draft",      "none",     "internal"): 0,
}

def get_authority_score(status: str, policy_authority: str, audience: str) -> int:
    """Return authority score for a document based on its front matter."""
    key = (status.lower(), policy_authority.lower(), audience.lower())
    if key in AUTHORITY_SCORES:
        return AUTHORITY_SCORES[key]
    if audience.lower() == "internal":
        return 0
    if status.lower() == "draft" or policy_authority.lower() == "none":
        return 0
    if status.lower() == "superseded":
        return 20
    return 50


def is_customer_facing(doc_meta: dict) -> bool:
    """Return True if a document is allowed to be used to answer customer queries."""
    status = doc_meta.get("status", "unknown").lower()
    audience = doc_meta.get("audience", "customer").lower()
    policy_authority = doc_meta.get("policy_authority", "none").lower()

    if status in EXCLUDED_FROM_RETRIEVAL:
        return False
    if audience in EXCLUDED_AUDIENCE:
        return False
    if policy_authority in EXCLUDED_POLICY_AUTHORITY:
        return False
    return True
