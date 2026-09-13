# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Validation evaluation and the gating decision (paper §3.2.4).

The harness never scores anything itself. An :class:`Evaluator` runs a skill
set on the validation split and returns an :class:`Evaluation`; a :class:`Gate`
compares the candidate with the best-so-far and decides. The default gate is
the paper's rule: accept iff the validation score strictly improves, stop early
at a perfect score. :class:`PairedGate` is the recommended production gate: it
needs per-task outcomes and accepts only when a paired bootstrap says the
improvement is unlikely to be noise. Hosts with richer statistics (safety
floors, cost policies) supply their own gate.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Protocol

from .skills import SkillSet
from .traces import TaskOutcome


@dataclass
class Evaluation:
    ref: str | None  # host-opaque identity of the evaluated skill set (digest, bundle id, ...)
    score: float | None  # R(T_val); None when the host's gate does not use a scalar
    aggregate: dict[str, Any] = field(default_factory=dict)
    per_task: list[TaskOutcome] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ref": self.ref, "score": self.score, "aggregate": self.aggregate, "tasks": len(self.per_task)}

    def to_document(self) -> dict[str, Any]:
        """Full form, including per-task outcomes, for resume checkpoints."""
        return {"ref": self.ref, "score": self.score, "aggregate": self.aggregate, "per_task": [t.to_dict() for t in self.per_task]}

    @classmethod
    def from_document(cls, value: dict[str, Any]) -> Evaluation:
        return cls(
            ref=value.get("ref"),
            score=value.get("score"),
            aggregate=dict(value.get("aggregate", {})),
            per_task=[TaskOutcome(**t) for t in value.get("per_task", [])],
        )


@dataclass
class Decision:
    accepted: bool
    feedback: dict[str, Any] = field(default_factory=dict)  # aggregates only; shown to the proposer next iteration
    stop: bool = False  # e.g. perfect validation score reached


class Evaluator(Protocol):
    async def evaluate(self, ref: str | None, skill_set: SkillSet, *, iteration: int, purpose: str) -> Evaluation:
        """``purpose`` is ``"baseline"`` for S_0 / the loaded head and ``"candidate"`` for S'_k."""
        ...


class Gate(Protocol):
    async def decide(self, best: Evaluation, candidate: Evaluation) -> Decision:
        """Called with ``best is candidate`` once before the loop as a perfect-score probe: answer ``accepted=False``
        and set ``stop`` from the score alone."""
        ...


class StrictImprovementGate:
    """Accept iff R(T_val,k) > R_best (Algorithm 1 line 13); stop when R_best reaches ``perfect``."""

    def __init__(self, *, perfect: float | None = 1.0, minimum_delta: float = 0.0):
        self.perfect, self.minimum_delta = perfect, minimum_delta

    async def decide(self, best: Evaluation, candidate: Evaluation) -> Decision:
        if best.score is None or candidate.score is None:
            raise ValueError("StrictImprovementGate needs scalar scores; supply a custom Gate otherwise")
        accepted = candidate.score > best.score + self.minimum_delta
        new_best = candidate.score if accepted else best.score
        stop = self.perfect is not None and new_best >= self.perfect
        return Decision(
            accepted=accepted,
            feedback={"best_score": round(best.score, 4), "candidate_score": round(candidate.score, 4), "accepted": accepted},
            stop=stop,
        )


class PairedGate:
    """Accept only when a paired bootstrap over per-task scores says the candidate is very likely better.

    The paper's rule (any increase of the mean) accepts noise about half the time on a few hundred validation
    tasks, and every noisy acceptance becomes the bar the next candidate must beat. This gate pairs the two
    evaluations by ``task_id``, resamples the paired differences ``resamples`` times and accepts iff the mean
    difference is positive and the fraction of resamples with a positive mean is at least
    ``min_win_probability``. It refuses to decide on fewer than ``min_tasks`` shared tasks.
    """

    def __init__(
        self,
        *,
        min_win_probability: float = 0.9,
        min_tasks: int = 20,
        resamples: int = 2000,
        seed: int = 17,
        perfect: float | None = 1.0,
    ):
        if not 0.5 <= min_win_probability <= 1.0:
            raise ValueError("min_win_probability must be in [0.5, 1.0]")
        self.min_win_probability, self.min_tasks, self.resamples, self.seed, self.perfect = (
            min_win_probability, min_tasks, resamples, seed, perfect,
        )

    async def decide(self, best: Evaluation, candidate: Evaluation) -> Decision:
        if best is candidate:  # the harness's perfect-score probe; refs are host-opaque and may both be None
            stop = self.perfect is not None and best.score is not None and best.score >= self.perfect
            return Decision(accepted=False, feedback={}, stop=stop)
        before = {t.task_id: t.score for t in best.per_task}
        after = {t.task_id: t.score for t in candidate.per_task}
        shared = sorted(set(before) & set(after))
        if len(shared) < self.min_tasks:
            return Decision(
                accepted=False,
                feedback={
                    "accepted": False,
                    "reason": "insufficient_paired_tasks",
                    "paired_tasks": len(shared),
                    "min_tasks": self.min_tasks,
                    "detail": "PairedGate needs per_task outcomes on both evaluations, paired by task_id",
                },
            )
        deltas = [after[t] - before[t] for t in shared]
        mean_delta = statistics.fmean(deltas)
        rng = random.Random(f"{self.seed}:{best.ref}:{candidate.ref}")
        n = len(deltas)
        wins, means = 0, []
        for _ in range(self.resamples):
            total = sum(deltas[rng.randrange(n)] for _ in range(n))
            means.append(total / n)
            wins += total > 0
        means.sort()
        win_probability = wins / self.resamples
        low, high = means[int(0.025 * (self.resamples - 1))], means[int(0.975 * (self.resamples - 1))]
        accepted = mean_delta > 0 and win_probability >= self.min_win_probability
        new_best = candidate.score if accepted else best.score
        stop = self.perfect is not None and new_best is not None and new_best >= self.perfect
        return Decision(
            accepted=accepted,
            feedback={
                "accepted": accepted,
                "paired_tasks": n,
                "mean_delta": round(mean_delta, 4),
                "delta_95_ci": [round(low, 4), round(high, 4)],
                "win_probability": round(win_probability, 3),
                "min_win_probability": self.min_win_probability,
                "best_score": None if best.score is None else round(best.score, 4),
                "candidate_score": None if candidate.score is None else round(candidate.score, 4),
            },
            stop=stop,
        )


__all__ = ["Decision", "Evaluation", "Evaluator", "Gate", "PairedGate", "StrictImprovementGate"]
