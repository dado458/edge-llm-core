"""
BaseWorker — background process for autonomous scheduled actions.

Subclass this to implement domain-specific follow-up logic:

    class SalesWorker(BaseWorker):
        def get_due_entities(self) -> list[dict]:
            ...  # scan memory for leads with pending followups
        def process_entity(self, entity: dict) -> None:
            ...  # run agent with a follow-up message
"""
import logging
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseWorker(ABC):
    """
    Polls for entities that need autonomous action and processes them.

    Usage:
        worker = MyWorker(agent=my_agent, memory=mem)
        worker.run_loop(interval_seconds=3600)   # blocking
        # or: worker.run_once()                  # one-shot (Railway Cron)
    """

    @abstractmethod
    def get_due_entities(self) -> list[dict]:
        """
        Return a list of entity dicts that need processing right now.
        Each dict must contain at least {"entity_id": str, "tenant_id": str}.
        Additional context fields (e.g. followup_context) are passed to
        process_entity as-is.
        """

    @abstractmethod
    def process_entity(self, entity: dict) -> None:
        """
        Execute the autonomous action for this entity.
        Typically calls agent.run(tenant_id, entity_id, trigger_message).
        """

    def on_start(self) -> None:
        """Called once before the first iteration. Override for setup."""

    def on_error(self, entity: dict, exc: Exception) -> None:
        """Called when process_entity raises. Override for alerting."""
        logger.error("Worker error on entity %s: %s", entity.get("entity_id"), exc)

    def run_once(self) -> int:
        """Process all due entities once. Returns number processed."""
        due = self.get_due_entities()
        count = 0
        for entity in due:
            try:
                self.process_entity(entity)
                count += 1
            except Exception as e:
                self.on_error(entity, e)
        return count

    def run_loop(self, interval_seconds: int = 3600) -> None:
        """Block and run indefinitely, sleeping between iterations."""
        self.on_start()
        logger.info("Worker started — interval %ds", interval_seconds)
        while True:
            try:
                n = self.run_once()
                logger.info("Worker processed %d entities", n)
            except Exception as e:
                logger.error("Worker iteration failed: %s", e)
            time.sleep(interval_seconds)
