# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Algorithm 1 (paper App. A) with the production extensions catalogued in docs/paper-differences.md.

    S_0 <- initial skills (or the stored head), W_0 <- stored wiki (or empty)
    R_best <- evaluate(S_0)
    for k in 1..K:
        T_k      <- trace_source.collect(S_{k-1})              # rollouts, or labelled production runs
        sample   <- stratified_sample(T_k)                      # <=5 failing, <=3 passing
        W'_k     <- maintainer.update(W_{k-1}, sample)          # never rolled back
        W'_k     <- pruner.prune(W'_k)                          # only when due (§3.6)
        P_k      <- proposer.propose(W'_k, S_{k-1}, T_k)        # ReAct, one atomic change
        S'_k     <- apply(S_{k-1}, P_k); eval <- evaluator.evaluate(S'_k)   # skipped for a known-rejected candidate (§3.8)
        accept iff gate.decide(best, eval); update skill-impact; persist W_k and (if accepted) S_k
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from .config import EvolveConfig
from .gates import Evaluation, Evaluator, Gate, StrictImprovementGate
from .hooks import BudgetExceeded, BudgetMeter, HookedModel, Hooks, IterationReport
from .model import ChatModel
from .redaction import Redactor, redact_trace
from .roles.maintainer import WikiMaintainer
from .roles.proposer import SkillProposer
from .roles.pruner import WikiPruner
from .skills import Proposal, ProposalError, SkillDocument, SkillSet, apply_proposal, unified_diff
from .stores.base import CheckpointStore, NullCheckpointStore, NullTraceStore, SkillStore, TraceStore, WikiStore
from .traces import Trace, TraceSource, stratified_sample
from .wiki import ImpactEntry, WikiState


@dataclass
class RunReport:
    iterations: int
    accepted: int
    rejected: int
    stopped_reason: str
    skills_ref: str | None
    wiki_ref: str | None
    skills: SkillSet
    wiki: WikiState
    best: Evaluation | None  # None only when the run stopped before the baseline evaluation (budget_exhausted)
    iteration_reports: list[IterationReport] = field(default_factory=list)
    model_calls: int = 0
    budget: dict[str, Any] = field(default_factory=dict)
    resumed_iteration: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "stopped_reason": self.stopped_reason,
            "skills_ref": self.skills_ref,
            "wiki_ref": self.wiki_ref,
            "best": self.best.to_dict() if self.best is not None else None,
            "model_calls": self.model_calls,
            "budget": self.budget,
            "resumed_iteration": self.resumed_iteration,
        }


async def evolve(
    *,
    config: EvolveConfig,
    model: ChatModel,
    wiki_store: WikiStore,
    skill_store: SkillStore,
    trace_source: TraceSource,
    evaluator: Evaluator,
    gate: Gate | None = None,
    trace_store: TraceStore | None = None,
    checkpoint_store: CheckpointStore | None = None,
    hooks: Hooks | None = None,
    initial_skills: list[SkillDocument] | None = None,
    baseline: Evaluation | None = None,
    redact: Redactor | None = None,
) -> RunReport:
    """Run one bounded evolution (up to ``config.max_iterations`` iterations) and persist the results.

    ``baseline`` lets a host that already evaluated the current head skip the baseline stage. ``checkpoint_store``
    enables resume: an iteration interrupted after its proposal was recorded is finished, not repeated. ``redact`` is
    applied to every collected trace's text before the roles or the trace store see it (§3.10).
    """
    hooks = hooks or Hooks()
    gate = gate or StrictImprovementGate()
    trace_store = trace_store or NullTraceStore()
    checkpoints = checkpoint_store or NullCheckpointStore()
    meter = BudgetMeter(config.budget)
    hooked = HookedModel(model, hooks, meter)
    maintainer, proposer, pruner = WikiMaintainer(hooked, config), SkillProposer(hooked, config), WikiPruner(hooked, config)

    # -- state -------------------------------------------------------------------------------------------------------
    skills_ref, skills = await skill_store.load()
    if skills_ref is None and initial_skills:
        skills = SkillSet.from_documents(initial_skills)
        seed = getattr(skill_store, "seed", None)
        if seed is not None:
            skills_ref = await seed(skills)
    wiki_ref, wiki = await wiki_store.load()
    invalidated = await wiki_store.invalidated_trace_ids()
    if invalidated:
        wiki = wiki.mark_stale(invalidated)
        await hooks.on_warning(f"{len(invalidated)} trace(s) invalidated by the host; affected citations marked stale")

    # -- baseline (Algorithm 1 line 2) ---------------------------------------------------------------------------------
    if baseline is None:
        try:
            meter.check(about_to="evaluation")
        except BudgetExceeded as exc:  # a zero budget stops cleanly, like exhaustion inside the loop
            await hooks.on_warning(str(exc))
            return RunReport(0, 0, 0, "budget_exhausted", skills_ref, wiki_ref, skills, wiki, None, [str(exc)], hooked.calls, meter.to_dict())
        async with hooks.stage("baseline"):
            baseline = await evaluator.evaluate(skills_ref, skills, iteration=0, purpose="baseline")
        meter.evaluations += 1
    best = baseline
    if not wiki.impact:
        wiki.record_impact(
            ImpactEntry(
                iteration=0,
                outcome="onboarded" if skills.skills else "baseline",
                action="baseline",
                target=None,
                diff=unified_diff(SkillSet(), skills) if skills.skills else "",
                validation=best.to_dict(),
                rationale=f"{len(skills)} onboarded skill(s)" if skills.skills else "empty skill set",
                candidate_digest=skills.content_digest(),
            )
        )
    stop_check = await gate.decide(best, best)
    if stop_check.stop:
        wiki_ref = await wiki_store.save(wiki, expected_ref=wiki_ref, iteration=wiki.iteration)
        return RunReport(0, 0, 0, "baseline_already_perfect", skills_ref, wiki_ref, skills, wiki, best, [], hooked.calls, meter.to_dict())

    # -- resume (§3.7): an iteration whose proposal was recorded but never finished is completed, not repeated -----
    resume: dict[str, Any] | None = None
    if wiki.iteration > 0:
        recorded = await checkpoints.load(wiki.iteration)
        if recorded and not recorded.get("completed") and "proposal" in recorded:
            resume = recorded

    accepted_total, rejected_streak, rejected_total = 0, 0, 0
    reports: list[IterationReport] = []
    stopped = "max_iterations"
    start = wiki.iteration if resume else wiki.iteration + 1
    feedback: dict[str, Any] = {}

    for k in range(start, start + config.max_iterations):
        warnings: list[str] = []
        try:
            meter.check(about_to="iteration")
        except BudgetExceeded as exc:
            await hooks.on_warning(str(exc))
            stopped = "budget_exhausted"
            break
        checkpoint = resume if (resume is not None and k == start) else None
        outcome, decision, diff, candidate_ref, candidate_digest, layer = "no_action", None, "", None, None, None
        proposal = Proposal(action="no_action", rationale="budget exhausted before the proposal stage")
        candidate_set: SkillSet | None = None
        checkpointed = False  # a completion marker is written only for iterations that recorded a proposal checkpoint
        sample: list[Trace] = []
        try:
            if checkpoint is None:
                # -- inference traces (line 7-8) --
                traces = await trace_source.collect(skills, iteration=k)
                if redact is not None:
                    traces = [redact_trace(t, redact) for t in traces]
                await trace_store.put(traces, iteration=k)
                sample, seen = stratified_sample(
                    traces, failing=config.failing_sample, passing=config.passing_sample, seen=set(wiki.seen_trace_ids), seed=config.seed, iteration=k
                )
                # -- wiki maintenance (line 9) --
                async with hooks.stage(f"wiki:{k}"):
                    wiki, maintainer_warnings = await maintainer.update(wiki, sample, skills, iteration=k, feedback=feedback)
                warnings.extend(maintainer_warnings)
                wiki.iteration, wiki.seen_trace_ids = k, sorted(seen)
                # -- pruning (§3.6), only when due --
                if wiki.patterns and _prune_due(config, wiki, k):
                    async with hooks.stage(f"prune:{k}"):
                        wiki, prune_warnings = await pruner.prune(wiki, iteration=k)
                    warnings.extend(prune_warnings)
                # -- skill proposal (line 10) --
                rejected = {e.candidate_digest: e.iteration for e in wiki.impact if e.outcome == "rejected" and e.candidate_digest}
                async with hooks.stage(f"proposal:{k}"):
                    proposal, _pad = await proposer.propose(wiki, skills, traces, iteration=k, feedback=feedback, rejected_candidates=rejected)
                if proposal.action != "no_action":
                    try:
                        candidate_set = apply_proposal(skills, proposal, iteration=k, allow_protected=config.allow_protected_edits)
                    except ProposalError as exc:  # the proposer validated a dry run, so this only fires if skills moved underneath
                        warnings.append(f"proposal could not be applied: {exc}")
                        proposal.rationale = f"{proposal.rationale} [not applied: {exc}]"
            else:
                proposal = Proposal.from_dict(checkpoint["proposal"], allow_promotions=True)  # validated when recorded
                candidate_set = SkillSet.from_document(checkpoint["candidate_set"])
                warnings.append(f"resumed iteration {k} from its checkpoint; the wiki and proposal stages were not repeated")

            if candidate_set is not None:
                candidate_digest = candidate_set.content_digest()
                layer = _changed_layer(skills, candidate_set, proposal)
                diff = unified_diff(skills, candidate_set)
                known = None if proposal.action == "promote" else _known_candidate(wiki, candidate_digest, skills)
                if proposal.action == "promote":
                    # A promotion moves a skill outward without changing what the agent reads, so validation cannot
                    # distinguish it from the incumbent; it is a structural change the host accepts or refuses (§3.2).
                    try:
                        candidate = await skill_store.propose(skills_ref, skills, proposal, candidate_set, iteration=k, wiki_ref=wiki_ref)
                        skills_ref = await skill_store.accept(candidate, expected_ref=skills_ref)
                    except ProposalError as exc:
                        warnings.append(f"host rejected the promotion: {exc}")
                        outcome, rejected_total, rejected_streak = "rejected", rejected_total + 1, rejected_streak + 1
                        feedback = {"accepted": False, "reason": "host_rejected_candidate", "detail": str(exc)}
                    else:
                        skills, candidate_ref = candidate.skill_set, candidate.ref
                        outcome, accepted_total, rejected_streak = "accepted", accepted_total + 1, 0
                        feedback = {"accepted": True, "reason": "structural_promotion", "validation": "not_required"}
                elif known is not None:  # §3.8: never pay to validate a candidate already judged
                    outcome, rejected_total, rejected_streak = "rejected", rejected_total + 1, rejected_streak + 1
                    feedback = {"accepted": False, **known}
                    warnings.append(f"candidate refused without validation: {known['reason']}")
                else:
                    if checkpoint is None:
                        wiki_ref = await wiki_store.save(wiki, expected_ref=wiki_ref, iteration=k)  # W'_k persists before validation
                        # The MODULE's candidate is recorded once and never overwritten: a resume re-materialises it through
                        # the host exactly as the first attempt did, so a host that normalises is never fed its own output.
                        record = {"proposal": proposal.to_dict(), "candidate_set": candidate_set.to_document(), "wiki_ref": wiki_ref}
                        await checkpoints.save(k, record)
                    else:
                        record = {key: checkpoint[key] for key in ("proposal", "candidate_set", "wiki_ref") if key in checkpoint}
                    checkpointed = True
                    try:
                        candidate = await skill_store.propose(skills_ref, skills, proposal, candidate_set, iteration=k, wiki_ref=wiki_ref)
                    except ProposalError as exc:  # the host's own validation (schema, taxonomy, ...) refused the candidate
                        warnings.append(f"host rejected the candidate: {exc}")
                        outcome, rejected_total, rejected_streak = "rejected", rejected_total + 1, rejected_streak + 1
                        feedback = {"accepted": False, "reason": "host_rejected_candidate", "detail": str(exc)}
                    else:
                        # The host's materialisation is canonical: it may re-key rows or add its own frontmatter, which is
                        # expected. What must not differ silently is what the agent reads: the instructions and overrides.
                        if _material(candidate.skill_set) != _material(candidate_set):
                            warnings.append("host materialised a candidate whose instructions differ from the proposal; continuing with the host's")
                        candidate_set = candidate.skill_set
                        diff = unified_diff(skills, candidate_set)
                        if checkpoint is not None and "evaluation" in checkpoint:
                            evaluation = Evaluation.from_document(checkpoint["evaluation"])
                            warnings.append("reused the recorded validation of the resumed candidate")
                        else:
                            meter.check(about_to="evaluation")
                            async with hooks.stage(f"validation:{k}"):
                                evaluation = await evaluator.evaluate(candidate.ref, candidate_set, iteration=k, purpose="candidate")
                            meter.evaluations += 1
                            await checkpoints.save(k, {**record, "evaluation": evaluation.to_document()})
                        decision = await gate.decide(best, evaluation)
                        candidate_ref = candidate.ref
                        if decision.accepted:
                            skills_ref = await skill_store.accept(candidate, expected_ref=skills_ref)
                            skills, best = candidate_set, evaluation
                            outcome, accepted_total, rejected_streak = "accepted", accepted_total + 1, 0
                        else:
                            outcome, rejected_total, rejected_streak = "rejected", rejected_total + 1, rejected_streak + 1
                        feedback = dict(decision.feedback)
        except BudgetExceeded as exc:
            warnings.append(str(exc))
            stopped = "budget_exhausted"
            if wiki.iteration != k:  # the maintainer did not finish; nothing of this iteration is persisted
                for message in warnings:
                    await hooks.on_warning(message)
                break
            outcome = "budget_exhausted"
            feedback = {"accepted": False, "reason": "budget_exhausted", "detail": str(exc)}
        if outcome == "no_action":  # a genuine no_action, or a proposal that could not be applied: no progress either way
            rejected_streak += 1
            reason = "no_action" if proposal.action == "no_action" else "proposal_not_applied"
            feedback = {"accepted": False, "reason": reason, "rationale": proposal.rationale}
        # -- skill-impact (line 18) --
        wiki.record_impact(
            ImpactEntry(
                iteration=k,
                outcome=outcome,
                action=proposal.action,
                target=proposal.name,
                diff=diff,
                validation={**(decision.feedback if decision else feedback if outcome in ("rejected", "accepted") else {}), "candidate_ref": candidate_ref},
                rationale=proposal.rationale,
                proposal_body=proposal.rendered_body() if outcome == "rejected" else None,
                citations=[c for c in proposal.citations if c in wiki.patterns],
                candidate_digest=candidate_digest,
                layer=layer,
            )
        )
        wiki_ref = await wiki_store.save(wiki, expected_ref=wiki_ref, iteration=k)
        if checkpointed:
            await checkpoints.save(k, {"completed": True})
        report = IterationReport(
            iteration=k,
            sampled_trace_ids=[t.id for t in sample],
            wiki_ref=wiki_ref,
            proposal=proposal.to_dict(),
            accepted=outcome == "accepted",
            outcome=outcome,
            feedback=feedback,
            warnings=warnings,
        )
        reports.append(report)
        for message in warnings:
            await hooks.on_warning(message)
        await hooks.on_iteration(report)
        if stopped == "budget_exhausted":
            break
        if decision is not None and decision.stop:
            stopped = "perfect_validation_score"
            break
        if rejected_streak >= config.max_rejected_streak:
            stopped = "rejected_streak"
            break

    return RunReport(
        iterations=len(reports),
        accepted=accepted_total,
        rejected=rejected_total,
        stopped_reason=stopped,
        skills_ref=skills_ref,
        wiki_ref=wiki_ref,
        skills=skills,
        wiki=wiki,
        best=best,
        iteration_reports=reports,
        model_calls=hooked.calls,
        budget=meter.to_dict(),
        resumed_iteration=start if resume else None,
    )


def _prune_due(config: EvolveConfig, wiki: WikiState, iteration: int) -> bool:
    if config.prune_every and iteration % config.prune_every == 0:
        return True
    return bool(config.wiki_char_budget) and wiki.render_size() > config.wiki_char_budget


def _known_candidate(wiki: WikiState, candidate_digest: str, current: SkillSet) -> dict[str, Any] | None:
    """A candidate the loop already has an answer for: identical to the current head, or rejected earlier."""
    if candidate_digest == current.content_digest():
        return {"reason": "no_change", "detail": "the candidate is identical to the current skill set"}
    for entry in reversed(wiki.impact):
        if entry.candidate_digest == candidate_digest and entry.outcome == "rejected":
            earlier = {k: v for k, v in entry.validation.items() if k != "candidate_ref"}
            return {"reason": "duplicate_candidate", "duplicate_of_iteration": entry.iteration, "earlier_validation": earlier}
    return None


def _material(skill_set: SkillSet) -> dict[str, tuple[str, str | None]]:
    """The agent-visible substance of a skill set: per skill, its body and what it overrides."""
    return {name: (skill.body, skill.overrides) for name, skill in skill_set.skills.items()}


def _changed_layer(before: SkillSet, after: SkillSet, proposal: Proposal) -> str | None:
    """The layer a proposal landed in (multi-layer sets only, so single-layer renderings match the paper)."""
    if not (before.multi_layer or after.multi_layer):
        return None
    if proposal.name in after.skills:
        return after.skills[proposal.name].layer
    if proposal.name in before.skills:
        return before.skills[proposal.name].layer
    return None


def evolve_sync(**kwargs: Any) -> RunReport:
    """Convenience wrapper for scripts and notebooks without an event loop."""
    return asyncio.run(evolve(**kwargs))


__all__ = ["RunReport", "evolve", "evolve_sync"]
