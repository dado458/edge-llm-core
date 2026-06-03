"""
EdgeAgent — the core loop of any Edge LLM agent.

Subclass this for each vertical:
    class SalesAgent(EdgeAgent):
        def build_system_prompt(self, tenant_cfg, stage_context): ...
        def get_tools(self): ...
        def get_tool_map(self): ...
"""
import json
import os
from abc import ABC, abstractmethod

import anthropic

from .memory.base import AbstractMemoryStore
from .tenants.base import AbstractTenantStore, TenantConfig
from .usage.base import AbstractUsageTracker
from .state_machine import StateMachine, StageContext


_COMPACT_THRESHOLD = 20    # messages before compaction kicks in
_COMPACT_MODEL     = "claude-haiku-4-5-20251001"
_MAX_LOOP_ITERS    = 10    # safety: abort if Claude keeps calling tools without end_turn
_MAX_MESSAGE_CHARS = 8_000 # input guard: ~2k tokens, enough for any real support/sales message


class EdgeAgent(ABC):
    """
    Base class for all Edge LLM agents.

    Concrete subclasses must implement:
        - build_system_prompt(tenant_cfg, stage_context) -> str
        - get_tools() -> list[dict]          (Anthropic tool definitions)
        - get_tool_map() -> dict[str, callable]
        - get_state_machine() -> StateMachine

    Optionally override:
        - initial_entity_state() -> dict     (default state for new entities)
        - on_before_run(tenant_cfg, entity_id, message) -> None
        - on_after_run(tenant_cfg, entity_id, reply) -> None
    """

    def __init__(
        self,
        memory:  AbstractMemoryStore,
        tenants: AbstractTenantStore,
        tracker: AbstractUsageTracker,
        model:   str = "claude-opus-4-8",
    ):
        self._memory  = memory
        self._tenants = tenants
        self._tracker = tracker
        self._model   = model
        self._client  = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self._sm      = self.get_state_machine()
        self._sm.validate()

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def build_system_prompt(self, tenant_cfg: TenantConfig, stage_ctx: StageContext) -> str:
        """Construct the full system prompt for this tenant + stage combination."""

    @abstractmethod
    def get_tools(self) -> list[dict]:
        """Return Anthropic tool definitions for the domain tools."""

    @abstractmethod
    def get_tool_map(self) -> dict[str, callable]:
        """Return {tool_name: callable} mapping for execution."""

    @abstractmethod
    def get_state_machine(self) -> StateMachine:
        """Return the domain state machine instance."""

    # ── Optional hooks ────────────────────────────────────────────────────────

    def initial_entity_state(self) -> dict:
        return {"stage": self._sm.initial_stage(), "interactions": 0}

    def on_before_run(self, tenant_cfg: TenantConfig, entity_id: str, message: str) -> None:
        pass

    def on_after_run(self, tenant_cfg: TenantConfig, entity_id: str, reply: str) -> None:
        pass

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, tenant_id: str, entity_id: str, incoming_message: str) -> str:
        """
        Main entry point. Receives one message, runs the internal agent loop,
        returns the final text reply. All state is persisted automatically.
        """
        # Bug fix: unknown tenant returns a clean string instead of raising KeyError.
        try:
            tenant_cfg = self._tenants.get(tenant_id)
        except KeyError:
            return f"[Unknown tenant '{tenant_id}']"

        # Bug fix: reject oversized inputs before they hit the LLM.
        if len(incoming_message) > _MAX_MESSAGE_CHARS:
            return f"[Message too long: {len(incoming_message)} chars, limit {_MAX_MESSAGE_CHARS}]"

        if self._tracker.is_over_limit(tenant_id, tenant_cfg.plan):
            return f"[Usage limit reached for tenant '{tenant_id}' on plan '{tenant_cfg.plan}']"

        entity_state = self._memory.get_entity_state(entity_id) or self.initial_entity_state()
        stage        = entity_state.get("stage", self._sm.initial_stage())

        if self._sm.is_terminal(stage):
            return f"[{entity_id} is in terminal state {stage} — no action]"

        self.on_before_run(tenant_cfg, entity_id, incoming_message)

        stage_ctx     = self._sm.get_context(stage)
        system_prompt = self.build_system_prompt(tenant_cfg, stage_ctx)

        # Bug fix: reload any compact summary persisted from a previous run and inject it,
        # so historical context survives across calls after compaction.
        prior_summary = entity_state.get("compact_summary", "")
        if prior_summary:
            system_prompt = self._inject_summary(system_prompt, prior_summary)

        messages = self._memory.get_conversation(entity_id)
        messages.append({"role": "user", "content": incoming_message})

        if self._should_compact(messages):
            messages, new_summary = self._compact(messages, tenant_id, prior_summary=prior_summary)
            entity_state["compact_summary"] = new_summary
            # Rebuild prompt with the merged summary so this run also sees it cleanly.
            system_prompt = self._inject_summary(
                self.build_system_prompt(tenant_cfg, stage_ctx), new_summary
            )
            self._memory.save_conversation(entity_id, messages)

        reply = self._loop(
            tenant_id=tenant_id,
            entity_id=entity_id,
            system_prompt=system_prompt,
            messages=messages,
            entity_state=entity_state,
        )

        self.on_after_run(tenant_cfg, entity_id, reply)
        return reply

    # ── Internal loop ────────────────────────────────────────────────────────

    def _loop(self, tenant_id: str, entity_id: str,
              system_prompt: str, messages: list, entity_state: dict) -> str:
        tools    = self.get_tools()
        tool_map = self.get_tool_map()

        for _ in range(_MAX_LOOP_ITERS):
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=system_prompt,
                tools=tools,
                messages=messages,
            )

            self._tracker.record(
                tenant_id=tenant_id,
                model=self._model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

            # Serialize SDK objects to plain dicts so json.dumps works in save_conversation.
            messages.append({"role": "assistant", "content": self._serialize_content(response.content)})

            if response.stop_reason in ("end_turn", "max_tokens"):
                reply = self._extract_text(response.content)
                entity_state["interactions"] = entity_state.get("interactions", 0) + 1
                self._memory.save_conversation(entity_id, messages)
                self._memory.save_entity_state(entity_id, entity_state)
                return reply

            if response.stop_reason == "tool_use":
                results = self._execute_tools(response.content, tool_map, entity_state)
                messages.append({"role": "user", "content": results})

        # Reached iteration limit — save state and return whatever text we have.
        self._memory.save_conversation(entity_id, messages)
        self._memory.save_entity_state(entity_id, entity_state)
        return "[agent loop limit reached]"

    def _execute_tools(self, content: list, tool_map: dict, entity_state: dict) -> list:
        results = []
        for block in content:
            if block.type != "tool_use":
                continue
            fn = tool_map.get(block.name)
            if not fn:
                result = {"error": f"Tool '{block.name}' not found"}
            else:
                try:
                    result = fn(**block.input)
                except Exception as e:
                    result = {"error": str(e)}
            results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     json.dumps(result, ensure_ascii=False),
            })
        return results

    # ── Compaction ───────────────────────────────────────────────────────────

    @staticmethod
    def _should_compact(messages: list) -> bool:
        return len(messages) >= _COMPACT_THRESHOLD

    def _compact(self, messages: list, tenant_id: str = "__compact__",
                 prior_summary: str = "") -> tuple[list, str]:
        """Summarize old messages with Haiku, return (trimmed_messages, summary).

        prior_summary: summary from a previous compaction — prepended so the new
        summary merges all historical context, not just the current window.
        """
        parts = []
        if prior_summary:
            parts.append(f"[Earlier conversation summary: {prior_summary}]")
        parts.extend(
            f"{m['role'].upper()}: {m['content'] if isinstance(m['content'], str) else '[tool]'}"
            for m in messages[:-4]
        )
        history_text = "\n".join(parts)
        resp = self._client.messages.create(
            model=_COMPACT_MODEL,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": (
                    "Summarize this conversation concisely, preserving key facts, "
                    "decisions made, current state, and important context.\n\n"
                    f"{history_text}"
                ),
            }],
        )
        if hasattr(self, "_tracker") and self._tracker and hasattr(resp, "usage"):
            self._tracker.record(
                tenant_id=tenant_id,
                model=_COMPACT_MODEL,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
            )
        summary = resp.content[0].text if resp.content else ""
        return messages[-4:], summary

    @staticmethod
    def _inject_summary(system_prompt: str, summary: str) -> str:
        return f"{system_prompt}\n\n# Conversation Summary (earlier messages)\n{summary}"

    @staticmethod
    def _serialize_content(content: list) -> list:
        """Convert Anthropic SDK content blocks to plain dicts for JSON serialization."""
        result = []
        for block in content:
            if isinstance(block, dict):
                result.append(block)
            elif hasattr(block, "model_dump"):
                result.append(block.model_dump())
            else:
                result.append(vars(block))
        return result

    @staticmethod
    def _extract_text(content: list) -> str:
        for block in content:
            if hasattr(block, "text"):
                return block.text
            if isinstance(block, dict) and block.get("type") == "text":
                return block["text"]
        return ""
