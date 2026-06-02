import json
from pathlib import Path

from .base import AbstractUsageTracker, UsageSummary


class LocalUsageTracker(AbstractUsageTracker):
    """JSON file-based usage tracker for dev / single-node deployments."""

    def __init__(self, data_dir: str | Path = "data/usage"):
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def record(self, tenant_id: str, model: str,
               input_tokens: int, output_tokens: int) -> None:
        month = self._current_month()
        data = self._load(tenant_id)
        entry = data.setdefault(month, {
            "calls": 0, "input_tokens": 0, "output_tokens": 0,
            "total_tokens": 0, "cost_usd": 0.0, "by_model": {}
        })
        cost = self._cost(model, input_tokens, output_tokens)
        entry["calls"]         += 1
        entry["input_tokens"]  += input_tokens
        entry["output_tokens"] += output_tokens
        entry["total_tokens"]  += input_tokens + output_tokens
        entry["cost_usd"]       = round(entry["cost_usd"] + cost, 6)
        m = entry["by_model"].setdefault(model, {"calls": 0, "cost_usd": 0.0})
        m["calls"]    += 1
        m["cost_usd"]  = round(m["cost_usd"] + cost, 6)
        self._save(tenant_id, data)

    def summary(self, tenant_id: str) -> UsageSummary:
        month = self._current_month()
        entry = self._load(tenant_id).get(month, {})
        return UsageSummary(
            tenant_id=tenant_id, month=month,
            calls=entry.get("calls", 0),
            input_tokens=entry.get("input_tokens", 0),
            output_tokens=entry.get("output_tokens", 0),
            total_tokens=entry.get("total_tokens", 0),
            cost_usd=entry.get("cost_usd", 0.0),
            by_model=entry.get("by_model", {}),
        )

    def get_all(self, tenant_id: str) -> list[UsageSummary]:
        return [
            UsageSummary(tenant_id=tenant_id, month=m, **v)
            for m, v in sorted(self._load(tenant_id).items())
        ]

    def _load(self, tenant_id: str) -> dict:
        path = self._dir / f"{tenant_id}.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, tenant_id: str, data: dict) -> None:
        path = self._dir / f"{tenant_id}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
