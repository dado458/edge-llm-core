import json

from .base import AbstractMemoryStore


class RedisMemoryStore(AbstractMemoryStore):
    """
    Redis-backed store for production multi-instance deployments.
    Uses a synchronous connection pool — safe for both sync and async
    callers (FastAPI async routes run sync I/O in a thread pool via
    anyio.to_thread; MCP tool handlers should do the same if needed).
    Requires: pip install edge-llm-core[redis]
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0", ttl_seconds: int = 86400 * 90):
        try:
            import redis as redis_lib
        except ImportError:
            raise ImportError("Install redis support: pip install 'edge-llm-core[redis]'")
        # connection_pool ensures thread-safe reuse across workers
        self._r = redis_lib.from_url(redis_url, decode_responses=True, max_connections=20)
        self._ttl = ttl_seconds

    def _conv_key(self, entity_id: str) -> str:
        return f"edge:conv:{entity_id}"

    def _state_key(self, entity_id: str) -> str:
        return f"edge:state:{entity_id}"

    def get_conversation(self, entity_id: str) -> list[dict]:
        raw = self._r.get(self._conv_key(entity_id))
        return json.loads(raw) if raw else []

    def save_conversation(self, entity_id: str, messages: list[dict]) -> None:
        self._r.set(self._conv_key(entity_id), json.dumps(messages, ensure_ascii=False), ex=self._ttl)

    def get_entity_state(self, entity_id: str) -> dict | None:
        raw = self._r.get(self._state_key(entity_id))
        return json.loads(raw) if raw else None

    def save_entity_state(self, entity_id: str, state: dict) -> None:
        self._r.set(self._state_key(entity_id), json.dumps(state, ensure_ascii=False), ex=self._ttl)
