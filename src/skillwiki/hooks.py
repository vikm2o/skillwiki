# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Host hooks: make paid stages idempotent, meter model calls, observe iterations.

Every method is optional; subclass and override what you need.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .config import Budget
from .model import ChatModel, ModelRequest, ModelResponse


class BudgetExceeded(Exception):
    """A :class:`skillwiki.config.Budget` ceiling was reached. The harness catches this and stops the run cleanly."""

    def __init__(self, limit: str, used: float, ceiling: float):
        super().__init__(f"budget exhausted: {limit} used {used} of {ceiling}")
        self.limit, self.used, self.ceiling = limit, used, ceiling


@dataclass
class IterationReport:
    iteration: int
    sampled_trace_ids: list[str]
    wiki_ref: str | None
    proposal: dict[str, Any]
    accepted: bool
    outcome: str
    feedback: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class Hooks:
    @contextlib.asynccontextmanager
    async def stage(self, name: str) -> AsyncIterator[None]:
        """Wraps one paid stage (``baseline``, ``wiki:k``, ``prune:k``, ``proposal:k``, ``validation:k``)."""
        yield

    async def on_model_call(self, role: str, request: ModelRequest, response: ModelResponse) -> None:
        return None

    async def on_iteration(self, report: IterationReport) -> None:
        return None

    async def on_warning(self, message: str) -> None:
        return None


@dataclass
class BudgetMeter:
    """Running totals checked against a :class:`Budget`. Shared by the hooked model and the harness."""

    budget: Budget
    started: float = field(default_factory=time.monotonic)
    model_calls: int = 0
    output_tokens: int = 0
    evaluations: int = 0

    @property
    def seconds(self) -> float:
        return time.monotonic() - self.started

    def check(self, *, about_to: str = "") -> None:
        """Raise :class:`BudgetExceeded` if any ceiling is already reached. ``about_to`` names the next paid step,
        so a stage is refused before it spends rather than after."""
        b = self.budget
        if b.max_model_calls is not None and self.model_calls >= b.max_model_calls:
            raise BudgetExceeded("max_model_calls", self.model_calls, b.max_model_calls)
        if b.max_output_tokens is not None and self.output_tokens >= b.max_output_tokens:
            raise BudgetExceeded("max_output_tokens", self.output_tokens, b.max_output_tokens)
        if b.max_evaluations is not None and about_to == "evaluation" and self.evaluations >= b.max_evaluations:
            raise BudgetExceeded("max_evaluations", self.evaluations, b.max_evaluations)
        if b.max_seconds is not None and self.seconds >= b.max_seconds:
            raise BudgetExceeded("max_seconds", round(self.seconds, 1), b.max_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_calls": self.model_calls,
            "output_tokens": self.output_tokens,
            "evaluations": self.evaluations,
            "seconds": round(self.seconds, 1),
            "ceilings": {
                "max_model_calls": self.budget.max_model_calls,
                "max_output_tokens": self.budget.max_output_tokens,
                "max_evaluations": self.budget.max_evaluations,
                "max_seconds": self.budget.max_seconds,
            },
        }


class HookedModel:
    """Wraps a ChatModel so every call is reported to ``hooks.on_model_call`` and counted against the budget.

    The budget is checked BEFORE each call, so a ReAct loop cannot overshoot by a whole proposal; the check raises
    :class:`BudgetExceeded`, which the harness turns into a clean ``budget_exhausted`` stop.
    """

    def __init__(self, model: ChatModel, hooks: Hooks, meter: BudgetMeter | None = None):
        self.model, self.hooks = model, hooks
        self.meter = meter or BudgetMeter(Budget())

    @property
    def calls(self) -> int:
        return self.meter.model_calls

    @property
    def supports_tools(self) -> bool:
        return bool(getattr(self.model, "supports_tools", False))

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.meter.check(about_to="model_call")
        response = await self.model.complete(request)
        self.meter.model_calls += 1
        usage = response.usage or {}
        tokens = usage.get("output_tokens", usage.get("completion_tokens"))  # Anthropic / OpenAI names
        if isinstance(tokens, int | float):
            self.meter.output_tokens += int(tokens)
        await self.hooks.on_model_call(request.role, request, response)
        return response


__all__ = ["BudgetExceeded", "BudgetMeter", "HookedModel", "Hooks", "IterationReport"]
