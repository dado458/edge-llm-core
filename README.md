# edge-llm-core

![Tests](https://github.com/dado458/edge-llm-core/actions/workflows/test.yml/badge.svg)

> **⚠️ ALPHA SOFTWARE — FOR DEVELOPMENT USE ONLY — v0.1.0**
>
> This package is intended for **development, prototyping, and research purposes only**.
> It is not suitable for production environments without significant additional hardening.
>
> - APIs may change without notice between minor versions in the 0.x series.
> - This package has not been audited for security. Do not use it to process, store, or transmit
>   sensitive, personal, or regulated data (PII, financial, health) without your own thorough security review.
> - All LLM calls made through this framework consume Anthropic API credits. Costs are your responsibility — monitor usage actively.
> - There is no guarantee of uptime, correctness, or fitness for any particular purpose.
> - **The authors accept no responsibility for any costs, data loss, security breaches, compliance violations,
>   or damages of any kind arising from the use or misuse of this software.**
> - Use entirely at your own risk.

**Framework for building autonomous Edge LLM agents as MCP servers.**

`edge-llm-core` defines the *Edge LLM Pattern* — a way to turn an MCP server from a
stateless API wrapper into an autonomous, stateful agent with its own internal LLM loop,
domain state machine, persistent memory, and multi-tenant configuration.

## The pattern in one sentence

> Instead of Claude calling an MCP tool to *get data*, Claude calls an MCP tool to *delegate
> an entire domain of responsibility* to a specialised agent that reasons, acts, and remembers
> autonomously.

## Install

> **Note:** PyPI publication is planned for v1.0.0 (stable release).
> Until then, install directly from GitHub:

```bash
# Current (GitHub)
pip install git+https://github.com/dado458/edge-llm-core.git

# With Redis support
pip install "edge-llm-core[redis] @ git+https://github.com/dado458/edge-llm-core.git"

# With FastAPI
pip install "edge-llm-core[server] @ git+https://github.com/dado458/edge-llm-core.git"
```

> Once published to PyPI (v1.0.0), installation will simplify to `pip install edge-llm-core`.

## Build a vertical in 4 steps

```python
# 1 — Define the domain state machine
from edge_llm.core.state_machine import StateMachine, StageContext

class SupportPipeline(StateMachine):
    stages          = ["OPEN", "IN_PROGRESS", "WAITING", "RESOLVED", "CLOSED"]
    terminal_stages = ["RESOLVED", "CLOSED"]
    transitions     = {
        "OPEN":        ["IN_PROGRESS", "CLOSED"],
        "IN_PROGRESS": ["WAITING", "RESOLVED", "CLOSED"],
        "WAITING":     ["IN_PROGRESS", "CLOSED"],
        "RESOLVED":    ["CLOSED"],
        "CLOSED":      [],
    }

# 2 — Implement the agent
from edge_llm.core.agent import EdgeAgent

class SupportAgent(EdgeAgent):
    def get_state_machine(self):
        return SupportPipeline()

    def build_system_prompt(self, tenant_cfg, stage_ctx):
        return f"You are a support agent for {tenant_cfg.name}. Current stage: {stage_ctx.stage}."

    def get_tools(self):
        return [...]  # Anthropic tool definitions

    def get_tool_map(self):
        return {"triage_ticket": triage_fn, "escalate": escalate_fn}

# 3 — Expose as MCP server
from edge_llm.core.mcp_server import BaseEdgeMCPServer
from edge_llm.core.memory.local import LocalMemoryStore
from edge_llm.core.tenants.local import LocalTenantStore
from edge_llm.core.usage.local import LocalUsageTracker

class SupportMCPServer(BaseEdgeMCPServer):
    SERVER_NAME = "support-agent"
    def create_agent(self):   return SupportAgent(self.create_memory(), self.create_tenants(), self.create_tracker())
    def create_memory(self):  return LocalMemoryStore()
    def create_tenants(self): return LocalTenantStore()
    def create_tracker(self): return LocalUsageTracker()

# 4 — Run
if __name__ == "__main__":
    SupportMCPServer().run()
```

Add to your `pyproject.toml`:
```toml
[project.scripts]
support-agent-mcp = "support_agent_mcp.server:main"
```

Then in your Claude config:
```json
{
  "mcpServers": {
    "support": {
      "command": "uvx",
      "args": ["support-agent-mcp"],
      "env": { "ANTHROPIC_API_KEY": "sk-ant-..." }
    }
  }
}
```

## What every vertical gets for free

`BaseEdgeMCPServer` automatically exposes 7 standard MCP tools:

| Tool | Description |
|---|---|
| `handle_message` | Send a message → runs full internal agent loop → returns reply |
| `get_entity_state` | Current state-machine stage + metadata |
| `get_conversation` | Message history (last N) |
| `set_entity_stage` | Manually advance or reset stage |
| `add_note` | Attach internal note without triggering the loop |
| `get_usage` | Monthly calls, tokens, cost for a tenant |
| `list_entities` | All active entities in memory |

## Architecture

```
Claude (central)
    │  delegates entire domain
    ▼
BaseEdgeMCPServer          ← your MCP server
    └── EdgeAgent           ← internal LLM loop (Claude Opus/Sonnet)
         ├── StateMachine   ← domain state (OPEN → IN_PROGRESS → RESOLVED)
         ├── Memory         ← persistent per-entity state + conversation
         ├── TenantStore    ← per-tenant config (name, prompt, plan)
         └── UsageTracker   ← billing-ready usage metering
```

## Available verticals

| Package | Domain | Install |
|---|---|---|
| [sales-agent-mcp](https://github.com/dado458/sales-agent-mcp) | Sales funnel (COLD→WON) | `pip install git+https://github.com/dado458/sales-agent-mcp.git` |
| [support-agent-mcp](https://github.com/dado458/support-agent-mcp) | Ticket triage + resolution | `pip install git+https://github.com/dado458/support-agent-mcp.git` |
| finance-agent-mcp | Invoice lifecycle | coming soon |

## Production backends

| Layer | Dev (default) | Production |
|---|---|---|
| Memory | `LocalMemoryStore` (JSON) | `RedisMemoryStore` |
| Tenants | `LocalTenantStore` (JSON) | subclass for PostgreSQL |
| Usage | `LocalUsageTracker` (JSON) | subclass for PostgreSQL |

Switch with environment variables — zero code changes.

## Known limitations

These are documented behaviours in v0.1.0 — not bugs, but things to be aware of before deploying:

| Limitation | Detail |
|---|---|
| **No API retry** | If the Anthropic API returns an error or times out mid-loop, the exception propagates uncaught. The MCP server will return an error to the host. Implement retries in your `on_before_run` / `on_after_run` hooks or wrap `agent.run()` at the call site. |
| **`max_tokens=1024` per loop iteration** | Each internal LLM call is capped at 1 024 output tokens. If the agent needs to produce a longer reply, the response will be truncated silently with `stop_reason: max_tokens`. Override `_MAX_TOKENS` by subclassing `EdgeAgent` if your domain needs longer outputs. |
| **`LocalMemoryStore` not thread-safe** | JSON file reads/writes are not atomic. Use `RedisMemoryStore` for any multi-instance or concurrent deployment. |
| **`list_entities` only works with `LocalMemoryStore`** | Production memory stores should override `_list_entities()` in their `BaseEdgeMCPServer` subclass. |
| **`schedule_followup` requires a worker** | The tool writes a pending flag in memory but does not trigger any action by itself. A `BaseWorker` subclass must be running separately to process it. |

## License

Apache 2.0 — free for commercial use, attribution required.
