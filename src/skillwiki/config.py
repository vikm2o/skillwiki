# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Loop configuration. Defaults follow the paper (Alg. 1, App. C) where it states a number."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Budget:
    """Hard ceilings for one ``evolve`` run. ``None`` means unlimited.

    ``max_output_tokens`` counts ``ModelResponse.usage["output_tokens"]`` when the model reports it; a model that
    reports nothing cannot exhaust it. ``max_evaluations`` counts validation runs (the baseline included). When a
    ceiling is hit the run stops with ``stopped_reason="budget_exhausted"``; the wiki work already done is saved.
    """

    max_model_calls: int | None = None
    max_output_tokens: int | None = None
    max_evaluations: int | None = None
    max_seconds: float | None = None


@dataclass(frozen=True)
class EvolveConfig:
    task_description: str = "tasks in this workspace"
    max_iterations: int = 8
    max_rejected_streak: int = 3
    # Wiki Maintainer sampling (App. C): up to 5 failing + 3 passing traces, 15k characters each.
    failing_sample: int = 5
    passing_sample: int = 3
    trace_char_cap: int = 15_000
    # Media policy: images attached only in the turn that fetched them.
    images_per_read: int = 4
    images_per_run: int = 32
    maintainer_images: int = 8
    # Skill Proposer ReAct budget (App. D: roughly 10-20 turns) and the paper's "read at least 4 traces" rule.
    react_max_turns: int = 20
    react_min_trace_reads: int = 4
    scratchpad_char_cap: int = 60_000
    # skill-impact.md keeps the full content of the most recent rejections.
    impact_full_bodies: int = 5
    # One retry when a role returns output the harness cannot apply; the error is fed back verbatim.
    role_retries: int = 1
    max_tokens: int = 4096
    seed: int = 17
    allow_protected_edits: bool = False
    # Layers (docs/paper-differences.md §3.2): may the proposer move a skill one layer outward? Hosts that own the
    # outer layers (a shared catalogue, a tenant) usually gate promotion themselves and leave this off.
    allow_promotions: bool = False
    budget: Budget = field(default_factory=Budget)
    # Wiki pruning (docs/paper-differences.md §3.6). Off by default (the paper never prunes). ``prune_every`` runs the
    # pruner every N iterations; ``wiki_char_budget`` runs it whenever the rendered wiki exceeds that many characters.
    # The pruner is shown every pattern's statistics and at most ``prune_bodies_char_cap`` characters of page bodies.
    prune_every: int = 0
    wiki_char_budget: int | None = None
    prune_bodies_char_cap: int = 30_000
    workspace: str = "default"
