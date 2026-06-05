"""
ChromaDB-backed catalog store.

Uses chromadb.PersistentClient — data survives restarts, no external service needed.
Each tenant gets an isolated collection named  catalog_<tenant_id>.
Embedding: ChromaDB default (all-MiniLM-L6-v2, downloaded on first use, ~80 MB).

Requires: pip install chromadb
"""
import logging

from .base import AbstractCatalogStore

logger = logging.getLogger(__name__)


class ChromaCatalogStore(AbstractCatalogStore):

    def __init__(self, persist_dir: str = "data/catalog"):
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "ChromaDB not installed. Run: pip install chromadb"
            )
        self._client = chromadb.PersistentClient(path=persist_dir)

    # ── Collection helper ────────────────────────────────────────────────────

    def _col(self, tenant_id: str):
        # Sanitize tenant_id — ChromaDB collection names must be [a-zA-Z0-9_-]
        safe = tenant_id.replace(".", "_").replace("/", "_")
        return self._client.get_or_create_collection(name=f"catalog_{safe}")

    # ── AbstractCatalogStore ─────────────────────────────────────────────────

    def upsert(self, tenant_id: str, documents: list[str], ids: list[str]) -> int:
        if not documents:
            return 0
        col = self._col(tenant_id)
        # ChromaDB upsert: insert if not exists, replace if id already present.
        col.upsert(documents=documents, ids=ids)
        logger.info("Catalog upsert: tenant=%s chunks=%d", tenant_id, len(documents))
        return len(documents)

    def search(self, tenant_id: str, query: str, n_results: int = 3) -> list[str]:
        col = self._col(tenant_id)
        total = col.count()
        if total == 0:
            return []
        k = min(n_results, total)
        results = col.query(query_texts=[query], n_results=k)
        docs = results.get("documents", [[]])[0]
        return docs

    def clear(self, tenant_id: str) -> None:
        safe = tenant_id.replace(".", "_").replace("/", "_")
        try:
            self._client.delete_collection(f"catalog_{safe}")
            logger.info("Catalog cleared: tenant=%s", tenant_id)
        except Exception as exc:
            logger.warning("Could not clear catalog for %s: %s", tenant_id, exc)

    def count(self, tenant_id: str) -> int:
        return self._col(tenant_id).count()
