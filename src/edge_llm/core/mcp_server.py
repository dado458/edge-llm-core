"""
BaseEdgeMCPServer — scaffolding for exposing an EdgeAgent as an MCP server.

Provides the standard set of tools every Edge LLM vertical should expose,
plus the plumbing to connect them to the agent and memory layer.

Usage:
    from edge_llm.core import BaseEdgeMCPServer

    class SalesMCPServer(BaseEdgeMCPServer):
        def create_agent(self) -> EdgeAgent:
            return SalesAgent(memory=..., tenants=..., tracker=...)

        def extra_tools(self) -> list:
            return []   # domain-specific additional tools

    if __name__ == "__main__":
        SalesMCPServer().run()
"""
import asyncio
from abc import ABC, abstractmethod

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .agent import EdgeAgent
from .memory.base import AbstractMemoryStore
from .tenants.base import AbstractTenantStore
from .usage.base import AbstractUsageTracker, PLAN_LIMITS


class BaseEdgeMCPServer(ABC):
    """
    Wires an EdgeAgent to the MCP protocol over stdio.

    Standard tools provided automatically:
        handle_message      — main entry point, runs the agent loop
        get_entity_state    — current domain state of an entity
        get_conversation    — message history
        set_entity_stage    — manually advance/reset stage
        add_note            — attach a note without running the loop
        get_usage           — monthly usage for a tenant
        list_entities       — list active entities in memory

    Subclasses can add domain-specific tools via extra_tools().
    """

    SERVER_NAME    = "edge-llm-agent"
    SERVER_VERSION = "0.1.0"

    @abstractmethod
    def create_agent(self) -> EdgeAgent:
        """Instantiate and return the domain EdgeAgent."""

    @abstractmethod
    def create_memory(self) -> AbstractMemoryStore:
        """Return the memory store instance."""

    @abstractmethod
    def create_tenants(self) -> AbstractTenantStore:
        """Return the tenant store instance."""

    @abstractmethod
    def create_tracker(self) -> AbstractUsageTracker:
        """Return the usage tracker instance."""

    def extra_tools(self) -> list[tuple[Tool, callable]]:
        """
        Return additional domain-specific (Tool, handler) pairs.
        Handler signature: (arguments: dict) -> dict
        """
        return []

    # ── Bootstrap ────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the MCP server on stdio (blocking)."""
        self._memory  = self.create_memory()
        self._tenants = self.create_tenants()
        self._tracker = self.create_tracker()
        self._agent   = self.create_agent()
        self._server  = Server(self.SERVER_NAME)
        self._register_tools()

        async def _main():
            async with stdio_server() as (read_stream, write_stream):
                await self._server.run(
                    read_stream, write_stream,
                    self._server.create_initialization_options(),
                )
        asyncio.run(_main())

    # ── Tool registration ─────────────────────────────────────────────────────

    def _register_tools(self) -> None:
        standard = self._standard_tools()
        extra    = self.extra_tools()
        all_tools = standard + extra

        tool_defs  = [t for t, _ in all_tools]
        tool_index = {t.name: fn for t, fn in all_tools}

        @self._server.list_tools()
        async def list_tools():
            return tool_defs

        @self._server.call_tool()
        async def call_tool(name: str, arguments: dict):
            fn = tool_index.get(name)
            if not fn:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
            try:
                result = fn(arguments)
                import json
                return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {e}")]

    def _standard_tools(self) -> list[tuple[Tool, callable]]:
        return [
            (
                Tool(
                    name="handle_message",
                    description="Send a message to the agent and get a reply. Runs the full internal agent loop.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "tenant_id":  {"type": "string", "description": "Tenant workspace ID"},
                            "entity_id":  {"type": "string", "description": "Entity identifier (lead, ticket, etc.)"},
                            "message":    {"type": "string", "description": "Incoming message text"},
                        },
                        "required": ["tenant_id", "entity_id", "message"],
                    },
                ),
                lambda a: {"reply": self._agent.run(a["tenant_id"], a["entity_id"], a["message"])}
            ),
            (
                Tool(
                    name="get_entity_state",
                    description="Return the current domain state of an entity.",
                    inputSchema={
                        "type": "object",
                        "properties": {"entity_id": {"type": "string"}},
                        "required": ["entity_id"],
                    },
                ),
                lambda a: self._memory.get_entity_state(a["entity_id"]) or {}
            ),
            (
                Tool(
                    name="get_conversation",
                    description="Return the message history for an entity.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity_id": {"type": "string"},
                            "last_n":    {"type": "integer", "description": "Return only the last N messages", "default": 20},
                        },
                        "required": ["entity_id"],
                    },
                ),
                lambda a: self._memory.get_conversation(a["entity_id"])[-a.get("last_n", 20):]
            ),
            (
                Tool(
                    name="set_entity_stage",
                    description="Manually set the stage of an entity in the state machine.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity_id": {"type": "string"},
                            "stage":     {"type": "string"},
                        },
                        "required": ["entity_id", "stage"],
                    },
                ),
                lambda a: self._memory.update_entity_state(a["entity_id"], stage=a["stage"])
            ),
            (
                Tool(
                    name="add_note",
                    description="Attach an internal note to an entity without triggering the agent loop.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity_id": {"type": "string"},
                            "note":      {"type": "string"},
                        },
                        "required": ["entity_id", "note"],
                    },
                ),
                lambda a: self._append_note(a["entity_id"], a["note"])
            ),
            (
                Tool(
                    name="get_usage",
                    description="Return monthly usage stats for a tenant.",
                    inputSchema={
                        "type": "object",
                        "properties": {"tenant_id": {"type": "string"}},
                        "required": ["tenant_id"],
                    },
                ),
                lambda a: self._usage_with_limit(a["tenant_id"])
            ),
            (
                Tool(
                    name="list_entities",
                    description="List all entities that have at least one conversation.",
                    inputSchema={"type": "object", "properties": {}, "required": []},
                ),
                lambda _: self._list_entities()
            ),
        ]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _append_note(self, entity_id: str, note: str) -> dict:
        state = self._memory.get_entity_state(entity_id)
        notes = state.get("notes", [])
        notes.append(note)
        return self._memory.update_entity_state(entity_id, notes=notes)

    def _usage_with_limit(self, tenant_id: str) -> dict:
        try:
            tenant  = self._tenants.get(tenant_id)
            plan    = tenant.plan
        except KeyError:
            plan = "basic"
        summary = self._tracker.summary(tenant_id)
        limit   = PLAN_LIMITS.get(plan)
        return {
            "tenant_id":       tenant_id,
            "plan":            plan,
            "month":           summary.month,
            "calls":           summary.calls,
            "limit":           limit,
            "calls_remaining": (limit - summary.calls) if limit else "unlimited",
            "cost_usd":        summary.cost_usd,
            "total_tokens":    summary.total_tokens,
        }

    def _list_entities(self) -> list[dict]:
        # Works for LocalMemoryStore; production stores should override.
        try:
            from pathlib import Path
            data_dir = getattr(self._memory, "_dir", None)
            if data_dir is None:
                return []
            seen: set[str] = set()
            entities = []
            for f in Path(data_dir).glob("*_state.json"):
                eid = f.stem.replace("_state", "")
                if eid not in seen:
                    seen.add(eid)
                    state = self._memory.get_entity_state(eid)
                    entities.append({"entity_id": eid, "stage": state.get("stage", "?")})
            return entities
        except Exception:
            return []
