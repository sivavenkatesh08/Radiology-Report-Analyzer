"""
FAISS Vector Store Module
--------------------------
Builds and queries a FAISS index from medical knowledge documents.
Uses sentence-transformers for embedding generation.
"""

import numpy as np
import logging
import sys
import os

logger = logging.getLogger(__name__)

# ── Ensure project root is on sys.path ──
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ── Model + FAISS initialization ──
FAISS_AVAILABLE = False
EMBEDDER_AVAILABLE = False

faiss = None
embedder = None
index = None
documents = []
metadata = []

try:
    import faiss as _faiss
    faiss = _faiss
    FAISS_AVAILABLE = True
    logger.info("✅ FAISS loaded successfully.")
except ImportError:
    logger.warning("⚠️ FAISS not installed. RAG retrieval disabled. Install with: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    EMBEDDER_AVAILABLE = True
    logger.info("✅ Sentence-Transformer embedder loaded.")
except Exception as e:
    logger.warning(f"⚠️ Sentence-Transformer not available: {e}")


def build_index():
    """
    Build FAISS index from the medical knowledge base.
    Called once at startup. Stores document embeddings for fast retrieval.
    """
    global index, documents, metadata

    if not FAISS_AVAILABLE or not EMBEDDER_AVAILABLE:
        logger.warning("Cannot build index — FAISS or embedder unavailable.")
        return False

    try:
        from src.rag.knowledge_base import get_all_documents, get_all_metadata

        documents = get_all_documents()
        metadata = get_all_metadata()

        if not documents:
            logger.warning("No documents in knowledge base.")
            return False

        # Generate embeddings
        embeddings = embedder.encode(documents, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product on normalized = cosine
        index.add(embeddings)

        logger.info(f"✅ FAISS index built with {len(documents)} documents, dim={dimension}.")
        return True

    except Exception as e:
        logger.error(f"Failed to build FAISS index: {e}")
        return False


def retrieve(query, top_k=3):
    """
    Retrieve the most relevant medical knowledge for a given query.

    Args:
        query: The report text or clinical question.
        top_k: Number of documents to retrieve.

    Returns:
        list of dicts with keys: content, topic, severity, recommendation, score
    """
    if index is None or not FAISS_AVAILABLE or not EMBEDDER_AVAILABLE:
        return []

    try:
        query_embedding = embedder.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding).astype("float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = index.search(query_embedding, min(top_k, len(documents)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(documents):
                continue
            results.append({
                "content": documents[idx],
                "topic": metadata[idx]["topic"],
                "severity": metadata[idx]["severity"],
                "recommendation": metadata[idx]["recommendation"],
                "score": float(score),
            })

        return results

    except Exception as e:
        logger.error(f"RAG retrieval failed: {e}")
        return []


def is_available():
    """Check if RAG system is fully operational."""
    return FAISS_AVAILABLE and EMBEDDER_AVAILABLE and index is not None


# ── Auto-build index on module load ──
_index_built = build_index()
