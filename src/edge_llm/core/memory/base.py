from abc import ABC, abstractmethod


class AbstractMemoryStore(ABC):
    """
    Persistence layer for entity conversations and state.
    'entity' is generic — a lead in sales, a ticket in support, an invoice in finance.
    """

    @abstractmethod
    def get_conversation(self, entity_id: str) -> list[dict]:
        """Return the message history for this entity (Anthropic messages format)."""

    @abstractmethod
    def save_conversation(self, entity_id: str, messages: list[dict]) -> None:
        """Persist the full message history."""

    @abstractmethod
    def get_entity_state(self, entity_id: str) -> dict | None:
        """Return the domain state dict for this entity, or None if not found."""

    @abstractmethod
    def save_entity_state(self, entity_id: str, state: dict) -> None:
        """Persist the domain state dict."""

    def update_entity_state(self, entity_id: str, **fields) -> dict:
        """Merge fields into the existing state and persist. Returns updated state."""
        state = self.get_entity_state(entity_id) or {}
        state.update(fields)
        self.save_entity_state(entity_id, state)
        return state
