"""
agent/indexer.py
Parses all Markdown files in knowledge-base/, extracts YAML front matter,
splits content into heading-aware chunks, embeds them with a LOCAL sentence-transformer
(no API key needed), and stores in ChromaDB with full metadata for authority-based
filtering and reranking.
"""

import logging
from pathlib import Path
from typing import Any

import frontmatter
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from agent.config import (
    KNOWLEDGE_BASE_DIR,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEBUG,
    get_authority_score,
    is_customer_facing,
)

logger = logging.getLogger(__name__)

# ── Singleton embedding model (loaded once) ───────────────────────────────────
_embed_model: SentenceTransformer | None = None

def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _embed_model


def _parse_document(md_path: Path) -> dict[str, Any]:
    """
    Parse a knowledge-base Markdown file.
    Returns dict with meta (front matter), content (body), filename.
    """
    post = frontmatter.load(str(md_path))
    meta = dict(post.metadata)
    meta.setdefault("status", "unknown")
    meta.setdefault("policy_authority", "none")
    meta.setdefault("audience", "customer")
    meta.setdefault("document_id", md_path.stem)
    meta.setdefault("title", md_path.stem)
    meta["filename"] = md_path.name
    meta["authority_score"] = get_authority_score(
        meta["status"], meta["policy_authority"], meta["audience"]
    )
    meta["customer_facing"] = is_customer_facing(meta)
    return {"meta": meta, "content": post.content, "filename": md_path.name}


def _split_into_chunks(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Split document content into chunks, preserving heading context.
    Each chunk includes the heading hierarchy as part of the text.
    """
    content = doc["content"]
    meta = doc["meta"]
    filename = doc["filename"]

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,
    )

    try:
        header_docs = header_splitter.split_text(content)
    except Exception:
        header_docs = [type("D", (), {"page_content": content, "metadata": {}})()]

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = []
    for hdoc in header_docs:
        sub_chunks = char_splitter.split_text(hdoc.page_content)
        heading_context = " > ".join(
            v for v in [
                hdoc.metadata.get("h1", ""),
                hdoc.metadata.get("h2", ""),
                hdoc.metadata.get("h3", ""),
            ] if v
        )
        for chunk_text in sub_chunks:
            if not chunk_text.strip():
                continue
            chunk_meta = {
                "filename": filename,
                "document_id": str(meta.get("document_id", "")),
                "title": str(meta.get("title", "")),
                "status": str(meta.get("status", "unknown")),
                "policy_authority": str(meta.get("policy_authority", "none")),
                "audience": str(meta.get("audience", "customer")),
                "authority_score": int(meta.get("authority_score", 0)),
                "customer_facing": bool(meta.get("customer_facing", False)),
                "heading_context": heading_context,
                "effective_date": str(meta.get("effective_date", "")),
                "supersedes": str(meta.get("supersedes", "")),
                "superseded_by": str(meta.get("superseded_by", "")),
                "customer_answering": str(meta.get("customer_answering", "true")),
            }
            chunks.append({"text": chunk_text.strip(), "meta": chunk_meta})

    return chunks


def build_index(force_rebuild: bool = False) -> chromadb.Collection:
    """
    Build (or load) the ChromaDB collection from the knowledge-base directory.
    Uses local sentence-transformers for embedding — no API key required.

    Args:
        force_rebuild: If True, deletes and recreates the collection.

    Returns:
        The ChromaDB collection object.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

    if force_rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info("Deleted existing collection for rebuild.")
        except Exception:
            pass

    # Check if collection already has data
    try:
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        if count > 0 and not force_rebuild:
            logger.info(f"Using existing index with {count} chunks.")
            return collection
    except Exception:
        pass

    logger.info("Building knowledge base index (using local embeddings — no API needed)...")

    embed_model = _get_embed_model()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    md_files = sorted(KNOWLEDGE_BASE_DIR.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {KNOWLEDGE_BASE_DIR}")

    all_texts = []
    all_metas = []
    all_ids = []

    for md_path in md_files:
        doc = _parse_document(md_path)

        if DEBUG:
            logger.debug(
                f"Parsed {doc['filename']}: status={doc['meta']['status']}, "
                f"authority={doc['meta']['policy_authority']}, "
                f"audience={doc['meta']['audience']}, "
                f"customer_facing={doc['meta']['customer_facing']}"
            )

        chunks = _split_into_chunks(doc)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['filename']}::chunk_{i}"
            all_ids.append(chunk_id)
            all_texts.append(chunk["text"])
            all_metas.append(chunk["meta"])

    # Embed all chunks locally (sentence-transformers, no API cost)
    logger.info(f"Embedding {len(all_texts)} chunks locally...")
    vectors = embed_model.encode(all_texts, show_progress_bar=True, batch_size=32)
    vectors_list = [v.tolist() for v in vectors]

    # Store in ChromaDB in one batch
    collection.add(
        ids=all_ids,
        embeddings=vectors_list,
        documents=all_texts,
        metadatas=all_metas,
    )

    total = collection.count()
    logger.info(f"Index complete. Total chunks: {total}")
    return collection


def get_collection() -> chromadb.Collection:
    """Return the existing ChromaDB collection (must have been built first)."""
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    return client.get_collection(COLLECTION_NAME)
