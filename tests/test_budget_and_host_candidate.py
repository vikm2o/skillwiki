# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Budgets stop a run cleanly and the host's materialised candidate is canonical."""

import json

from skillwiki import (
    Candidate,
    Evaluation,
    EvolveConfig,
    Hooks,
    ScriptedChatModel,
    SkillDocument,
    SkillSet,
    StaticTraceSource,
    TaskOutcome,
    Trace,
    evolve,
)
from skillwiki.config import Budget
from skillwiki.stores import MemoryRevisionStore, RevisionSkillStore, RevisionWikiStore

SKILL = "---\nname: rule\ndescription: r\n---\n# Rule\nDo the thing.\n"


def _traces():
    return [Trace(id=f"t{i}", outcome=TaskOutcome(task_id=f"task{i}", score=float(i >= 6), passed=i >= 6), text=f"body {i}") for i in range(10)]


def _tool(name, **args):
    return json.dumps({"tool": name, "args": args})


def _maintainer_reply(ids):
    return json.dumps({"create_patterns": [{"name": "p", "content": "# P\nx", "evidence": ids[:1]}], "update_patterns": [],
                       "update_index": "- [p](wiki/patterns/p.md): P; cause; fix", "append_log": "log"})


def _model(proposal):
    def script(request):
        if request.role == "wiki_maintainer":
            ids = [line.split()[2] for line in request.text.splitlines() if line.startswith("=== Trace ")]
            return _maintainer_reply(ids)
        turns = request.text.count("### Turn ")
        if turns < 4:
            return _tool("read_trace", id=f"t{turns}")
        return _tool("finish", proposal=proposal)
    return ScriptedChatModel(script)


class Scorer:
    def __init__(self):
        self.seen = []

    async def evaluate(self, ref, skill_set, *, iteration, purpose):
        self.seen.append((purpose, sorted(skill_set.skills), skill_set.skills.get("rule", SkillDocument("rule", "")).body))
        return Evaluation(ref=ref, score=0.5 if purpose == "baseline" else 0.9)


async def test_budget_stops_the_run_before_the_next_paid_call_and_reports_usage():
    revisions = MemoryRevisionStore()
    warnings = []

    class Warn(Hooks):
        async def on_warning(self, message):
            warnings.append(message)

    proposal = {"action": "create", "name": "rule", "skill_md": SKILL, "purpose_md": "## Origin\nx", "citations": ["p"]}
    # the maintainer costs one call; the proposer needs five -> the budget trips inside the proposal stage
    report = await evolve(
        config=EvolveConfig(max_iterations=3, budget=Budget(max_model_calls=3)),
        model=_model(proposal),
        wiki_store=RevisionWikiStore(revisions, "ws"),
        skill_store=RevisionSkillStore(revisions, "ws"),
        trace_source=StaticTraceSource(_traces()),
        evaluator=Scorer(),
        hooks=Warn(),
    )
    assert report.stopped_reason == "budget_exhausted" and report.model_calls == 3 and report.iterations == 1
    assert report.budget["model_calls"] == 3 and report.budget["ceilings"]["max_model_calls"] == 3
    assert report.iteration_reports[0].outcome == "budget_exhausted" and any("max_model_calls" in w for w in warnings)
    # the maintainer's work from that iteration is persisted, and the impact log says why nothing was proposed
    _ref, wiki = await RevisionWikiStore(revisions, "ws").load()
    assert "p" in wiki.patterns and wiki.impact[-1].outcome == "budget_exhausted" and wiki.iteration == 1


async def test_evaluation_budget_refuses_the_validation_stage():
    revisions = MemoryRevisionStore()
    proposal = {"action": "create", "name": "rule", "skill_md": SKILL, "purpose_md": "## Origin\nx", "citations": ["p"]}
    scorer = Scorer()
    report = await evolve(
        config=EvolveConfig(max_iterations=2, budget=Budget(max_evaluations=1)),  # the baseline uses the only evaluation
        model=_model(proposal),
        wiki_store=RevisionWikiStore(revisions, "ws"),
        skill_store=RevisionSkillStore(revisions, "ws"),
        trace_source=StaticTraceSource(_traces()),
        evaluator=scorer,
    )
    assert report.stopped_reason == "budget_exhausted" and [p for p, _s, _b in scorer.seen] == ["baseline"]
    assert report.budget["evaluations"] == 1 and report.wiki.impact[-1].outcome == "budget_exhausted"


async def test_host_materialised_candidate_is_what_gets_validated_and_kept():
    """A host may normalise the candidate (QC collapses an override into a successor row). The harness validates and
    continues with the host's version, warns about the difference, and records the module's content digest."""
    revisions = MemoryRevisionStore()
    inner = RevisionSkillStore(revisions, "ws")
    warnings = []

    class Warn(Hooks):
        async def on_warning(self, message):
            warnings.append(message)

    class NormalisingStore:
        async def load(self):
            return await inner.load()

        async def propose(self, current_ref, current, proposal, candidate, *, iteration, wiki_ref):
            normalised = candidate.copy()
            normalised.skills["rule"].body = normalised.skills["rule"].body.upper()
            return Candidate(ref="host-1", skill_set=normalised, proposal=proposal)

        async def accept(self, candidate, *, expected_ref):
            return await inner.accept(candidate, expected_ref=expected_ref)

    proposal = {"action": "create", "name": "rule", "skill_md": SKILL, "purpose_md": "## Origin\nx", "citations": ["p", "not-a-pattern"]}
    scorer = Scorer()
    report = await evolve(
        config=EvolveConfig(max_iterations=1),
        model=_model(proposal),
        wiki_store=RevisionWikiStore(revisions, "ws"),
        skill_store=NormalisingStore(),
        trace_source=StaticTraceSource(_traces()),
        evaluator=scorer,
        hooks=Warn(),
    )
    assert report.accepted == 1 and report.skills.skills["rule"].body == "# RULE\nDO THE THING.\n"
    assert scorer.seen[-1][2] == "# RULE\nDO THE THING.\n"  # validated the host's version, not the module's
    assert any("instructions differ from the proposal" in w for w in warnings)
    entry = report.wiki.impact[-1]
    module_digest = SkillSet.from_documents([SkillDocument.parse_skill_md(SKILL)]).content_digest()
    assert entry.candidate_digest == module_digest and entry.citations == ["p"] and entry.layer is None
    assert "+DO THE THING." in entry.diff
