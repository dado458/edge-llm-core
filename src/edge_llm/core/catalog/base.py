from abc import ABC, abstractmethod


class AbstractCatalogStore(ABC):
    """
    Per-tenant vector store for product knowledge base.
    Each tenant has an isolated namespace (collection).
    """

    @abstractmethod
    def upsert(self, tenant_id: str, documents: list[str], ids: list[str]) -> int:
        """Embed and store/replace documents. Returns number stored."""

    @abstractmethod
    def search(self, tenant_id: str, query: str, n_results: int = 3) -> list[str]:
        """Return the top-n most semantically similar document chunks."""

    @abstractmethod
    def clear(self, tenant_id: str) -> None:
        """Delete all documents for this tenant."""

    @abstractmethod
    def count(self, tenant_id: str) -> int:
        """Return the number of stored chunks for this tenant."""
