import json
from pathlib import Path

from .base import AbstractMemoryStore


class LocalMemoryStore(AbstractMemoryStore):
    """
    JSON file-based store for development and single-instance deployments.
    In production swap for RedisMemoryStore via the factory in your app.
    """

    def __init__(self, data_dir: str | Path = "data/memory"):
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def get_conversation(self, entity_id: str) -> list[dict]:
        path = self._dir / f"{entity_id}_conversation.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def save_conversation(self, entity_id: str, messages: list[dict]) -> None:
        path = self._dir / f"{entity_id}_conversation.json"
        path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_entity_state(self, entity_id: str) -> dict | None:
        path = self._dir / f"{entity_id}_state.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_entity_state(self, entity_id: str, state: dict) -> None:
        path = self._dir / f"{entity_id}_state.json"
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_entity_states(self) -> list[tuple[str, dict]]:
        result = []
        for f in self._dir.glob("*_state.json"):
            entity_id = f.stem.replace("_state", "")
            state = self.get_entity_state(entity_id)
            if state is not None:
                result.append((entity_id, state))
        return result
