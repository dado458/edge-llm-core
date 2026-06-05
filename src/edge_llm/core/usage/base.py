from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


# Cost per million tokens (USD) — updated for Claude 4.x pricing
MODEL_COSTS: dict[str, dict[str, float]] = {
    "claude-opus-4-8":          {"input": 15.0,  "output": 75.0},
    "claude-sonnet-4-6":        {"input": 3.0,   "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.25,  "output": 1.25},
}

# Monthly call limits per plan (None = unlimited)
PLAN_LIMITS: dict[str, int | None] = {
    "basic":      500,
    "pro":        5_000,
    "enterprise": None,
}

# Monthly total-token budgets per plan (input + output combined, None = unlimited).
# Calibrated at ~2× the expected usage for the call limit so normal use never hits
# the cap, while a single runaway conversation (giant context) gets blocked.
PLAN_TOKEN_LIMITS: dict[str, int | None] = {
    "basic":      1_000_000,    # ~2× 500 avg calls @ 1 100 tok each
    "pro":        10_000_000,   # ~2× 5 000 avg calls @ 1 100 tok each
    "enterprise": None,
}

# Hard cap on input tokens for a single call, by plan.
# Prevents one call from consuming the whole monthly budget.
MAX_INPUT_TOKENS_PER_CALL: dict[str, int] = {
    "basic":       8_000,   # ~6 back-and-forth turns + system prompt
    "pro":        32_000,   # longer conversations
    "enterprise": 200_000,  # nearly uncapped
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
        """True when the monthly call count has reached the plan cap."""
        limit = PLAN_LIMITS.get(plan)
        if limit is None:
            return False
        return self.summary(tenant_id).calls >= limit

    def is_over_token_limit(self, tenant_id: str, plan: str) -> bool:
        """True when cumulative monthly tokens have reached the plan token budget."""
        limit = PLAN_TOKEN_LIMITS.get(plan)
        if limit is None:
            return False
        return self.summary(tenant_id).total_tokens >= limit

    def would_exceed_token_limit(self, tenant_id: str, plan: str, estimated_tokens: int) -> bool:
        """True if adding estimated_tokens would breach the monthly token budget."""
        limit = PLAN_TOKEN_LIMITS.get(plan)
        if limit is None:
            return False
        return (self.summary(tenant_id).total_tokens + estimated_tokens) > limit

    @staticmethod
    def _current_month() -> str:
        return datetime.now().strftime("%Y-%m")

    @staticmethod
    def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
        rates = MODEL_COSTS.get(model, {"input": 3.0, "output": 15.0})
        return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
