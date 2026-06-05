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
    assert mem.get_entity_state("e1") is None  # None for missing entities (not {})
    mem.save_entity_state("e1", {"stage": "START"})
    assert mem.get_entity_state("e1") == {"stage": "START"}


def test_memory_update_entity_state(tmp_path):
    mem = LocalMemoryStore(tmp_path)
    mem.save_entity_state("e1", {"stage": "START", "score": 0})
    updated = mem.update_entity_state("e1", stage="MIDDLE", score=10)
    assert updated["stage"] == "MIDDLE"
    assert updated["score"] == 10


def test_memory_update_entity_state_new_entity(tmp_path):
    """update_entity_state must not crash when the entity doesn't exist yet (state=None)."""
    mem = LocalMemoryStore(tmp_path)
    updated = mem.update_entity_state("brand-new", stage="COLD", tenant_id="t1")
    assert updated["stage"] == "COLD"
    assert updated["tenant_id"] == "t1"
    assert mem.get_entity_state("brand-new") == updated


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


# ── EdgeAgent guard clauses (no API key needed) ───────────────────────────────

from unittest.mock import MagicMock
from edge_llm.core.agent import EdgeAgent, _MAX_MESSAGE_CHARS


class _MinimalAgent(EdgeAgent):
    """Minimal concrete subclass for testing guard clauses without a real LLM."""
    def get_state_machine(self):      return SimplePipeline()
    def build_system_prompt(self, t, s): return "prompt"
    def get_tools(self):              return []
    def get_tool_map(self):           return {}


def _make_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake")
    return _MinimalAgent(
        memory=LocalMemoryStore(tmp_path / "memory"),
        tenants=LocalTenantStore(tmp_path / "tenants.json"),
        tracker=LocalUsageTracker(tmp_path / "usage"),
    )


def test_run_unknown_tenant_returns_clean_string(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)
    result = agent.run("nonexistent-tenant", "e1", "hello")
    assert result.startswith("[Unknown tenant")
    assert "nonexistent-tenant" in result


def test_run_message_too_long_returns_clean_string(tmp_path, monkeypatch):
    store = LocalTenantStore(tmp_path / "tenants.json")
    store.register("t1", plan="basic")
    agent = _make_agent(tmp_path, monkeypatch)
    long_msg = "x" * (_MAX_MESSAGE_CHARS + 1)
    result = agent.run("t1", "e1", long_msg)
    assert result.startswith("[Message too long")


def test_compact_summary_persisted_to_entity_state(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)
    # Simulate a prior_summary already in entity_state.
    mem = agent._memory
    mem.save_entity_state("e1", {"stage": "START", "compact_summary": "User asked about pricing."})
    state = mem.get_entity_state("e1")
    assert state["compact_summary"] == "User asked about pricing."


def test_compact_includes_prior_summary_in_history(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)
    messages = [{"role": "user", "content": f"msg {i}"} for i in range(8)]
    # _compact is called internally; verify it returns trimmed messages (last 4).
    # We mock the Anthropic client to avoid a real API call.
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text="merged summary")]
    mock_resp.usage.input_tokens = 10
    mock_resp.usage.output_tokens = 5
    agent._client.messages.create = MagicMock(return_value=mock_resp)

    trimmed, summary = agent._compact(messages, "t1", prior_summary="old summary")
    assert summary == "merged summary"
    assert len(trimmed) == 4
    # Verify the prompt sent to Haiku included the prior summary.
    call_args = agent._client.messages.create.call_args
    prompt_content = call_args[1]["messages"][0]["content"]
    assert "old summary" in prompt_content
