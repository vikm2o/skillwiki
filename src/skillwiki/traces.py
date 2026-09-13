# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""The Raw Layer: execution traces and how they are sampled for the roles.

A :class:`Trace` is what the host recorded when the agent ran a task with the
current skills: a text rendering of the trajectory (reasoning, tool calls,
observations, final answer), optional media, and the scored outcome. The
:class:`TraceSource` protocol lets the host decide whether traces come from a
fresh rollout (the paper's setup) or from production runs that later received
a label (the same thing at zero extra cost).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Protocol

from .model import ImagePart
from .skills import SkillSet


@dataclass
class TaskOutcome:
    task_id: str
    score: float  # in [0, 1]; the paper's f(ŷ, y)
    passed: bool
    prediction: Any = None
    truth: Any = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "score": self.score,
            "passed": self.passed,
            "prediction": self.prediction,
            "truth": self.truth,
            "meta": self.meta,
        }


@dataclass
class Trace:
    id: str
    outcome: TaskOutcome
    text: str
    media: list[ImagePart] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)  # opaque host provenance, copied onto every citation of this trace
    skills_in_play: list[str] = field(default_factory=list)  # names of the skills the agent had in context for this task

    @property
    def task_id(self) -> str:
        return self.outcome.task_id

    def to_document(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "outcome": self.outcome.to_dict(),
            "text": self.text,
            "media": [m.to_dict() for m in self.media],
            "meta": self.meta,
            "skills_in_play": list(self.skills_in_play),
        }

    @classmethod
    def from_document(cls, value: dict[str, Any]) -> Trace:
        outcome = value["outcome"]
        return cls(
            id=value["id"],
            outcome=TaskOutcome(**outcome),
            text=value.get("text", ""),
            media=[ImagePart.from_dict(m) for m in value.get("media", [])],
            meta=dict(value.get("meta", {})),
            skills_in_play=list(value.get("skills_in_play", [])),
        )

    def rendered(self, *, char_cap: int) -> str:
        text = self.text
        if len(text) > char_cap:
            text = text[:char_cap] + f"\n[... truncated {len(self.text) - char_cap} characters ...]"
        status = "PASS" if self.outcome.passed else "FAIL"
        head = f"Trace {self.id} (task {self.task_id}) -- {status}, score {self.outcome.score:.3f}"
        if self.outcome.prediction is not None or self.outcome.truth is not None:
            head += f"\nprediction: {self.outcome.prediction!r}\nground truth: {self.outcome.truth!r}"
        if self.skills_in_play:
            head += "\nskills in play: " + ", ".join(self.skills_in_play)
        if self.media:
            head += f"\nattached media: {len(self.media)}"
        return f"{head}\n\n{text}"


class TraceSource(Protocol):
    async def collect(self, skill_set: SkillSet, *, iteration: int) -> list[Trace]: ...


class StaticTraceSource:
    """A fixed list of traces (useful for tests and for archived evidence)."""

    def __init__(self, traces: list[Trace]):
        self.traces = list(traces)

    async def collect(self, skill_set: SkillSet, *, iteration: int) -> list[Trace]:
        return list(self.traces)


def stratified_sample(
    traces: list[Trace],
    *,
    failing: int = 5,
    passing: int = 3,
    seen: set[str] | None = None,
    seed: int = 17,
    iteration: int = 0,
) -> tuple[list[Trace], set[str]]:
    """Paper App. C: up to ``failing`` failing and ``passing`` passing traces.

    When the trace set does not change between iterations, sampling rotates
    through unseen traces first so consecutive iterations do not hand the
    maintainer identical evidence; when a stratum is exhausted its ``seen`` set
    is reset rather than sampling fewer than the budget. Returns the sample and
    the updated ``seen`` set.
    """
    seen = set(seen or ())
    rng = random.Random(f"{seed}:{iteration}")
    sample: list[Trace] = []
    for wanted, predicate in ((failing, lambda t: not t.outcome.passed), (passing, lambda t: t.outcome.passed)):
        pool = [t for t in traces if predicate(t)]
        if not pool or wanted <= 0:
            continue
        unseen = [t for t in pool if t.id not in seen]
        if len(unseen) < min(wanted, len(pool)):
            for t in pool:
                seen.discard(t.id)
            unseen = pool
        chosen = rng.sample(unseen, min(wanted, len(unseen)))
        sample.extend(chosen)
        seen.update(t.id for t in chosen)
    return sample, seen


def summarize_outcomes(traces: list[Trace]) -> str:
    """The concise all-tasks summary handed to the proposer (paper §3.2.3): status, prediction, truth."""
    total = len(traces)
    passed = sum(t.outcome.passed for t in traces)
    lines = [f"Training outcomes: {passed}/{total} passed ({(passed / total if total else 0):.1%})", ""]
    for trace in sorted(traces, key=lambda t: (t.outcome.passed, t.id)):
        status = "PASS" if trace.outcome.passed else "FAIL"
        lines.append(
            f"- {trace.id} [{status}] score={trace.outcome.score:.2f}"
            + (f" prediction={_short(trace.outcome.prediction)} truth={_short(trace.outcome.truth)}"
               if trace.outcome.prediction is not None or trace.outcome.truth is not None else "")
        )
    return "\n".join(lines)


def summarize_skill_usage(traces: list[Trace]) -> str:
    """Per-skill usage over ``traces`` (§3.9): how often each skill was in play on failing and passing tasks.

    Only skills that appear in at least one trace are listed, so the table is bounded by what the agent actually saw,
    not by the catalogue. Returns an empty string when no trace names a skill.
    """
    usage: dict[str, list[int]] = {}
    for trace in traces:
        for name in trace.skills_in_play:
            counts = usage.setdefault(name, [0, 0])
            counts[0 if not trace.outcome.passed else 1] += 1
    if not usage:
        return ""
    lines = ["| skill | in play (failing) | in play (passing) |", "| --- | --- | --- |"]
    for name, (failing, passing) in sorted(usage.items(), key=lambda kv: (-kv[1][0], kv[0])):
        lines.append(f"| {name} | {failing} | {passing} |")
    return "\n".join(lines)


def _short(value: Any, limit: int = 80) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


__all__ = ["StaticTraceSource", "TaskOutcome", "Trace", "TraceSource", "stratified_sample", "summarize_outcomes", "summarize_skill_usage"]
