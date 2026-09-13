# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""The paper's loop, end to end, with a scripted model and the in-memory store.

Iteration 1: maintainer creates a pattern; proposer creates a skill that the evaluator scores LOWER -> rejected.
Iteration 2: maintainer adds evidence to the same pattern; proposer, having read skill-impact, proposes a different
skill -> accepted. Iteration 3: no_action. A second run on the same stores starts from the compounded wiki.
"""

import json

import pytest

from skillwiki import (
    Decision,
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
from skillwiki.stores import MemoryRevisionStore, RevisionSkillStore, RevisionTraceStore, RevisionWikiStore

SKILL_A = "---\nname: goal_directed\ndescription: abstract\n---\n# Goal directed\nThink about goals.\n"
SKILL_B = "---\nname: break_loop\ndescription: concrete rule\n---\n# Break loop\nNever return an item to its origin location.\n"


def _traces():
    return [
        Trace(id=f"t{i}", outcome=TaskOutcome(task_id=f"task{i}", score=0.0 if i < 6 else 1.0, passed=i >= 6, prediction="x", truth="y"), text=f"trace body {i}",
              meta={"label_revision_ids": [f"rev{i}"]})
        for i in range(10)
    ]


class ScoreByName:
    """Evaluator: a skill set scores by which skills it contains."""

    scores = {frozenset(): 0.5, frozenset({"goal_directed"}): 0.4, frozenset({"break_loop"}): 0.8}
    calls: list[tuple[str, str | None]] = []

    async def evaluate(self, ref, skill_set: SkillSet, *, iteration, purpose):
        self.calls.append((purpose, ref))
        score = self.scores[frozenset(skill_set.skills)]
        return Evaluation(ref=ref, score=score, aggregate={"accuracy": score}, per_task=[])


def _tool(name, **args):
    return json.dumps({"tool": name, "args": args})


def _maintainer(iteration, sample_ids):
    if iteration == 1:
        return json.dumps(
            {
                "create_patterns": [{"name": "take-examine-move-loop", "title": "Loop", "content": "# Loop\nAgent loops.", "evidence": sample_ids[:2]}],
                "update_patterns": [],
                "update_index": "- [take-examine-move-loop](wiki/patterns/take-examine-move-loop.md): loops; cause; fix",
                "append_log": "Found looping.",
            }
        )
    return json.dumps(
        {
            "create_patterns": [],
            "update_patterns": [{"name": "take-examine-move-loop", "edits": [{"op": "append", "content": f"Evidence iter {iteration}."}], "evidence": sample_ids[:1]}],
            "update_index": "- [take-examine-move-loop](wiki/patterns/take-examine-move-loop.md): loops; cause; fix (updated)",
            "append_log": f"Iteration {iteration}: loop persists.",
        }
    )


def make_model(state):
    """Scripted maintainer + proposer. The proposer reads index, impact, 4 traces, then finishes."""

    def script(request):
        text = request.text
        if request.role == "wiki_maintainer":
            iteration = int(text.split("# Iteration ")[1].split("\n")[0])
            ids = [line.split()[2] for line in text.splitlines() if line.startswith("=== Trace ")]
            state["maintainer_saw"].append(ids)
            return _maintainer(iteration, ids)
        # proposer
        iteration = int(text.split("# Iteration ")[1].split("\n")[0])
        turns = text.count("### Turn ")
        if turns == 0:
            return _tool("read_index")
        if turns == 1:
            return _tool("read_impact")
        if turns == 2:
            state["impact_seen"][iteration] = text
        if 2 <= turns < 6:
            return _tool("read_trace", id=f"t{turns - 2}")
        if iteration == 1:
            proposal = {"action": "create", "name": "goal_directed", "skill_md": SKILL_A, "purpose_md": "## Origin\nloop pattern", "rationale": "abstract", "citations": ["take-examine-move-loop"]}
        elif iteration == 2:
            proposal = {"action": "create", "name": "break_loop", "skill_md": SKILL_B, "purpose_md": "## Origin\nrejected goal_directed", "rationale": "concrete", "citations": ["take-examine-move-loop"]}
        else:
            proposal = {"action": "no_action", "rationale": "nothing new"}
        return _tool("finish", proposal=proposal)

    return ScriptedChatModel(script)


class Recording(Hooks):
    def __init__(self):
        self.stages, self.calls, self.reports = [], 0, []

    def stage(self, name):
        self.stages.append(name)
        return super().stage(name)

    async def on_model_call(self, role, request, response):
        self.calls += 1

    async def on_iteration(self, report):
        self.reports.append(report)


@pytest.fixture
def stores():
    revisions = MemoryRevisionStore()
    return revisions, RevisionWikiStore(revisions, "ws"), RevisionSkillStore(revisions, "ws"), RevisionTraceStore(revisions, "ws")


async def test_loop_compounds_the_wiki_and_gates_skills(stores):
    revisions, wiki_store, skill_store, trace_store = stores
    state = {"maintainer_saw": [], "impact_seen": {}}
    hooks = Recording()
    evaluator = ScoreByName()
    evaluator.calls = []
    report = await evolve(
        config=EvolveConfig(max_iterations=3, max_rejected_streak=3, react_min_trace_reads=4),
        model=make_model(state),
        wiki_store=wiki_store,
        skill_store=skill_store,
        trace_store=trace_store,
        trace_source=StaticTraceSource(_traces()),
        evaluator=evaluator,
        hooks=hooks,
    )
    assert (report.iterations, report.accepted, report.rejected, report.stopped_reason) == (3, 1, 1, "max_iterations")
    assert set(report.skills.skills) == {"break_loop"} and report.best.score == 0.8
    # wiki compounded: the pattern created at k=1 was patched at k=2 and k=3, and evidence accumulated
    page = report.wiki.patterns["take-examine-move-loop"]
    assert page.created_iteration == 1 and page.updated_iteration == 3 and "Evidence iter 2." in page.body and "Evidence iter 3." in page.body
    assert len(page.evidence) >= 3
    assert all(e.meta == {"label_revision_ids": [f"rev{e.trace_id[1:]}"]} for e in page.evidence)  # Trace.meta rides on citations
    # the maintainer saw different traces on consecutive iterations (rotation), not the same fixed rows
    assert state["maintainer_saw"][0] != state["maintainer_saw"][1]
    # skill-impact carried the rejection (with its full body) to the proposer at iteration 2
    assert "REJECTED" in state["impact_seen"][2] and "goal_directed" in state["impact_seen"][2] and "Think about goals." in state["impact_seen"][2]
    impact = [(e.iteration, e.action, e.outcome) for e in report.wiki.impact]
    assert impact == [(0, "baseline", "baseline"), (1, "create", "rejected"), (2, "create", "accepted"), (3, "no_action", "no_action")]
    assert report.wiki.impact[1].diff.startswith("--- /dev/null") and report.wiki.impact[1].proposal_body is not None
    # stages, persistence and the raw layer
    assert hooks.stages[:4] == ["baseline", "wiki:1", "proposal:1", "validation:1"] and "validation:3" not in hooks.stages
    assert [c[0] for c in evaluator.calls] == ["baseline", "candidate", "candidate"]
    wiki_chain = await revisions.list("ws", "wiki")
    assert len(wiki_chain) == 5 and wiki_chain[0].digest == report.wiki_ref  # W'_1, W_1, W'_2, W_2, W_3
    skills_chain = await revisions.list("ws", "skills")
    assert len(skills_chain) == 1 and skills_chain[0].digest == report.skills_ref
    assert (await trace_store.get("t3")) is not None
    assert hooks.calls == report.model_calls > 0

    # ---- a second run on the same stores starts from the compounded wiki and the accepted skills ----
    state2 = {"maintainer_saw": [], "impact_seen": {}}

    def script2(request):
        if request.role == "wiki_maintainer":
            assert "take-examine-move-loop" in request.text and "Evidence iter 3." in request.text
            ids = [line.split()[2] for line in request.text.splitlines() if line.startswith("=== Trace ")]
            return _maintainer(4, ids)
        turns = request.text.count("### Turn ")
        if turns == 0:
            assert "break_loop" in request.text  # active skills are the accepted head
            return _tool("read_index")
        if turns == 1:
            return _tool("read_impact")
        if turns == 2:
            state2["impact_seen"][4] = request.text
        return _tool("finish", proposal={"action": "no_action", "rationale": "stable"})

    report2 = await evolve(
        config=EvolveConfig(max_iterations=1),
        model=ScriptedChatModel(script2),
        wiki_store=wiki_store,
        skill_store=skill_store,
        trace_source=StaticTraceSource(_traces()),
        evaluator=evaluator,
    )
    assert report2.wiki.iteration == 4 and report2.wiki.patterns["take-examine-move-loop"].created_iteration == 1
    assert "Iteration 1: create goal_directed -- REJECTED" in state2["impact_seen"][4]
    assert [e.iteration for e in report2.wiki.impact] == [0, 1, 2, 3, 4]


async def test_onboarded_baseline_and_protected_skills(stores):
    _, wiki_store, skill_store, _ = stores
    seed = [SkillDocument(name="break_loop", body="rule", frontmatter={"description": "onboarded"}, protected=True)]

    def script(request):
        if request.role == "wiki_maintainer":
            return json.dumps({"create_patterns": [], "update_patterns": [], "update_index": "", "append_log": "nothing"})
        turns = request.text.count("### Turn ")
        if turns == 0:
            assert "break_loop (protected)" in request.text
            return _tool("finish", proposal={"action": "retire", "name": "break_loop", "rationale": "try"})
        assert "protected" in request.text  # the rejection came back as an observation, not an exception
        return _tool("finish", proposal={"action": "no_action", "rationale": "protected"})

    report = await evolve(
        config=EvolveConfig(max_iterations=1, react_min_trace_reads=0),
        model=ScriptedChatModel(script),
        wiki_store=wiki_store,
        skill_store=skill_store,
        trace_source=StaticTraceSource(_traces()),
        evaluator=ScoreByName(),
        initial_skills=seed,
    )
    assert report.wiki.impact[0].outcome == "onboarded" and "1 onboarded skill" in report.wiki.impact[0].rationale
    assert set(report.skills.skills) == {"break_loop"} and report.skills_ref is not None  # seeded as revision 0


async def test_perfect_baseline_stops_early(stores):
    _, wiki_store, skill_store, _ = stores

    class Perfect:
        async def evaluate(self, ref, skill_set, *, iteration, purpose):
            return Evaluation(ref=ref, score=1.0)

    report = await evolve(
        config=EvolveConfig(), model=ScriptedChatModel([]), wiki_store=wiki_store, skill_store=skill_store,
        trace_source=StaticTraceSource([]), evaluator=Perfect(),
    )
    assert report.stopped_reason == "baseline_already_perfect" and report.model_calls == 0


async def test_custom_gate_and_maintainer_retry(stores):
    _, wiki_store, skill_store, _ = stores
    attempts = {"maintainer": 0}

    class AlwaysReject:
        async def decide(self, best, candidate):
            return Decision(accepted=False, feedback={"reason": "host said no"})

    def script(request):
        if request.role == "wiki_maintainer":
            attempts["maintainer"] += 1
            if attempts["maintainer"] == 1:
                return "not json at all"
            assert "previous output was rejected" in request.text
            return json.dumps({"create_patterns": [], "update_patterns": [], "update_index": "", "append_log": "ok"})
        return _tool("finish", proposal={"action": "no_action", "rationale": "x"})

    report = await evolve(
        config=EvolveConfig(max_iterations=1, react_min_trace_reads=0), model=ScriptedChatModel(script), wiki_store=wiki_store,
        skill_store=skill_store, trace_source=StaticTraceSource(_traces()), evaluator=ScoreByName(), gate=AlwaysReject(),
    )
    assert attempts["maintainer"] == 2 and report.wiki.log[-1].text == "ok"
    assert any("rejected (attempt 1)" in w for w in report.iteration_reports[0].warnings)
