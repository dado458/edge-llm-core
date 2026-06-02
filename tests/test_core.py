"""
Basic smoke tests for edge-llm-core.
Run: pytest tests/
"""
import pytest
from edge_llm.core.state_machine import StateMachine, StageContext
from edge_llm.core.memory.local import LocalMemoryStore
from edge_llm.core.tenants.local import LocalTenantStore
from edge_llm.core.tenants.base import TenantConfig
from edge_llm.core.usage.local import LocalUsageTracker
from edge_llm.core.usage.base import PLAN_LIMITS


# ── StateMachine ─────────────────────────────────────────────────────────────

class SimplePipeline(StateMachine):
    stages          = ["START", "MIDDLE", "DONE", "CANCELLED"]
    terminal_stages = ["DONE", "CANCELLED"]
    transitions     = {
        "START":     ["MIDDLE", "CANCELLED"],
        "MIDDLE":    ["DONE", "CANCELLED"],
        "DONE":      [],
        "CANCELLED": [],
    }
    _context_map = {
        "START":  StageContext("START",  "Begin.", ["tool_a"], ["MIDDLE"]),
        "MIDDLE": StageContext("MIDDLE", "Continue.", ["tool_b"], ["DONE"]),
    }


def test_state_machine_validate():
    sm = SimplePipeline()
    sm.validate()  # should not raise


def test_state_machine_is_terminal():
    sm = SimplePipeline()
    assert sm.is_terminal("DONE")
    assert sm.is_terminal("CANCELLED")
    assert not sm.is_terminal("START")


def test_state_machine_transitions():
    sm = SimplePipeline()
    assert sm.can_transition("START", "MIDDLE")
    assert not sm.can_transition("START", "DONE")


def test_state_machine_context():
    sm = SimplePipeline()
    ctx = sm.get_context("START")
    assert ctx.stage == "START"
    assert "tool_a" in ctx.recommended_tools


def test_initial_stage():
    sm = SimplePipeline()
    assert sm.initial_stage() == "START"


# ── LocalMemoryStore ─────────────────────────────────────────────────────────

def test_memory_conversation(tmp_path):
    mem = LocalMemoryStore(tmp_path)
    assert mem.get_conversation("e1") == []
    msgs = [{"role": "user", "content": "hello"}]
    mem.save_conversation("e1", msgs)
    assert mem.get_conversation("e1") == msgs


def test_memory_entity_state(tmp_path):
    mem = LocalMemoryStore(tmp_path)
    assert mem.get_entity_state("e1") == {}
    mem.save_entity_state("e1", {"stage": "START"})
    assert mem.get_entity_state("e1") == {"stage": "START"}


def test_memory_update_entity_state(tmp_path):
    mem = LocalMemoryStore(tmp_path)
    mem.save_entity_state("e1", {"stage": "START", "score": 0})
    updated = mem.update_entity_state("e1", stage="MIDDLE", score=10)
    assert updated["stage"] == "MIDDLE"
    assert updated["score"] == 10


# ── LocalTenantStore ─────────────────────────────────────────────────────────

def test_tenant_register_and_get(tmp_path):
    store = LocalTenantStore(tmp_path / "tenants.json")
    key = store.register("acme", name="Acme Inc", plan="pro")
    assert key.startswith("sk-")
    cfg = store.get("acme")
    assert cfg.name == "Acme Inc"
    assert cfg.plan == "pro"
    assert cfg.active is True


def test_tenant_get_by_key(tmp_path):
    store = LocalTenantStore(tmp_path / "tenants.json")
    key = store.register("acme")
    found = store.get_by_key(key)
    assert found is not None
    assert found.tenant_id == "acme"


def test_tenant_not_found(tmp_path):
    store = LocalTenantStore(tmp_path / "tenants.json")
    with pytest.raises(KeyError):
        store.get("nonexistent")


def test_tenant_duplicate_raises(tmp_path):
    store = LocalTenantStore(tmp_path / "tenants.json")
    store.register("acme")
    with pytest.raises(ValueError):
        store.register("acme")


def test_tenant_meta(tmp_path):
    store = LocalTenantStore(tmp_path / "tenants.json")
    store.register("acme", meta={"agent_name": "Giulia", "language": "it"})
    cfg = store.get("acme")
    assert cfg.get("agent_name") == "Giulia"
    assert cfg.get("language") == "it"


# ── LocalUsageTracker ─────────────────────────────────────────────────────────

def test_usage_record_and_summary(tmp_path):
    tracker = LocalUsageTracker(tmp_path)
    tracker.record("acme", "claude-haiku-4-5-20251001", input_tokens=100, output_tokens=50)
    summary = tracker.summary("acme")
    assert summary.calls == 1
    assert summary.input_tokens == 100
    assert summary.output_tokens == 50
    assert summary.total_tokens == 150
    assert summary.cost_usd > 0


def test_usage_over_limit(tmp_path):
    tracker = LocalUsageTracker(tmp_path)
    limit = PLAN_LIMITS["basic"]  # 500
    for _ in range(limit):
        tracker.record("acme", "claude-haiku-4-5-20251001", 10, 5)
    assert tracker.is_over_limit("acme", "basic")
    assert not tracker.is_over_limit("acme", "pro")


def test_usage_enterprise_no_limit(tmp_path):
    tracker = LocalUsageTracker(tmp_path)
    for _ in range(10_000):
        tracker.record("big", "claude-haiku-4-5-20251001", 1, 1)
    assert not tracker.is_over_limit("big", "enterprise")
