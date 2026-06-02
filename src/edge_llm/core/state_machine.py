from dataclasses import dataclass, field


@dataclass
class StageContext:
    stage: str
    objective: str
    recommended_tools: list[str] = field(default_factory=list)
    possible_next_stages: list[str] = field(default_factory=list)


class StateMachine:
    """
    Domain-agnostic state machine for an Edge LLM agent.

    Subclass this to define the stages, transitions, and per-stage context
    for your domain (sales funnel, support ticket lifecycle, invoice flow, etc.).

    Example — sales funnel:
        class SalesPipeline(StateMachine):
            stages         = ["COLD", "INTERESTED", "OBJECTION", "CLOSING", "WON", "LOST"]
            terminal_stages = ["WON", "LOST"]
            transitions    = {
                "COLD":      ["INTERESTED", "LOST"],
                "INTERESTED":["OBJECTION", "CLOSING", "LOST"],
                ...
            }
    """

    # Subclasses must define these three class attributes:
    stages: list[str] = []
    terminal_stages: list[str] = []
    transitions: dict[str, list[str]] = {}

    # Optional per-stage metadata — subclasses override get_context() instead
    # if they want computed context.
    _context_map: dict[str, StageContext] = {}

    def get_context(self, stage: str) -> StageContext:
        """
        Return context for the agent's system prompt enrichment.
        Override in subclasses for computed/dynamic context.
        """
        if stage in self._context_map:
            return self._context_map[stage]
        return StageContext(
            stage=stage,
            objective=f"Handle the {stage} stage.",
            possible_next_stages=self.transitions.get(stage, []),
        )

    def is_terminal(self, stage: str) -> bool:
        return stage in self.terminal_stages

    def can_transition(self, from_stage: str, to_stage: str) -> bool:
        return to_stage in self.transitions.get(from_stage, [])

    def initial_stage(self) -> str:
        return self.stages[0] if self.stages else "START"

    def validate(self) -> None:
        """Sanity-check the machine definition at startup."""
        for stage in self.terminal_stages:
            if stage not in self.stages:
                raise ValueError(f"Terminal stage '{stage}' not in stages list")
        for src, targets in self.transitions.items():
            if src not in self.stages:
                raise ValueError(f"Transition source '{src}' not in stages list")
            for t in targets:
                if t not in self.stages:
                    raise ValueError(f"Transition target '{t}' not in stages list")
