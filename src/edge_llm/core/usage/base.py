from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


# Cost per million tokens (USD) — updated for Claude 4.x pricing
MODEL_COSTS: dict[str, dict[str, float]] = {
    "claude-opus-4-8":          {"input": 15.0,  "output": 75.0},
    "claude-sonnet-4-6":        {"input": 3.0,   "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.25,  "output": 1.25},
}

# Default call limits per plan (None = unlimited)
PLAN_LIMITS: dict[str, int | None] = {
    "basic":      500,
    "pro":        5_000,
    "enterprise": None,
}


@dataclass
class UsageSummary:
    tenant_id: str
    month: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    by_model: dict = field(default_factory=dict)


class AbstractUsageTracker(ABC):

    @abstractmethod
    def record(self, tenant_id: str, model: str,
               input_tokens: int, output_tokens: int) -> None:
        """Record a single LLM call for the current month."""

    @abstractmethod
    def summary(self, tenant_id: str) -> UsageSummary:
        """Return aggregated usage for the current month."""

    @abstractmethod
    def get_all(self, tenant_id: str) -> list[UsageSummary]:
        """Return full history, one entry per month."""

    def is_over_limit(self, tenant_id: str, plan: str) -> bool:
        limit = PLAN_LIMITS.get(plan)
        if limit is None:
            return False
        return self.summary(tenant_id).calls >= limit

    @staticmethod
    def _current_month() -> str:
        return datetime.now().strftime("%Y-%m")

    @staticmethod
    def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
        rates = MODEL_COSTS.get(model, {"input": 3.0, "output": 15.0})
        return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
