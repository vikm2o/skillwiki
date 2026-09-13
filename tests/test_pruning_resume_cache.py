# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Phase 2: wiki pruning, resume from checkpoints, and the evaluation cache."""

import json

import pytest

from skillwiki import (
    Evaluation,
    EvidenceRef,
    EvolveConfig,
    Hooks,
    ImpactEntry,
    Pattern,
    PruneUpdate,
    ScriptedChatModel,
    SkillDocument,
    SkillSet,
    StaticTraceSource,
    TaskOutcome,
    Trace,
    WikiError,
    WikiState,
    evolve,
)
from skillwiki.stores import MemoryRevisionStore, RevisionCheckpointStore, RevisionSkillStore, RevisionWikiStore

SKILL = "---\nname: rule\ndescription: r\n---\n# Rule\nDo the thing.\n"
SKILL_B = "---\nname: other\ndescription: o\n---\n# Other\nDo the other thing.\n"


def _wiki_with(*ids, accepted_citing=()):
    wiki = WikiState()
    for i, pid in enumerate(ids):
        wiki.patterns[pid] = Pattern(id=pid, title=pid.title(), body=f"# {pid}\nbody {pid}", evidence=[EvidenceRef(f"t{i}")],
                                     stale_evidence=[EvidenceRef(f"s{i}")], created_iteration=1, updated_iteration=1 + i)
    wiki.index = [__import__("skillwiki").IndexEntry(pid, "desc") for pid in ids]
    for pid in accepted_citing:
        wiki.record_impact(ImpactEntry(iteration=1, outcome="accepted", action="create", target="s", diff="", validation={}, citations=[pid]))
    return wiki


def test_apply_prune_merges_retires_and_protects():
    wiki = _wiki_with("loop", "loop-dup", "stale-only", "keep", accepted_citing=("keep",))
    update = PruneUpdate.from_dict({
        "merges": [{"into": "loop", "from": ["loop-dup"], "content": "# Loop\nmerged body", "title": "Loop (merged)"}],
        "retire": [{"name": "stale-only", "reason": "all evidence retracted"}],
        "append_log": "Merged the duplicate loop page; retired stale-only.",
    })
    pruned = wiki.apply_prune(update, iteration=5, bodies_shown={"loop"})
    assert set(pruned.patterns) == {"loop", "keep"} and set(pruned.retired) == {"loop-dup", "stale-only"}
    loop = pruned.patterns["loop"]
    assert loop.body == "# Loop\nmerged body" and loop.title == "Loop (merged)" and loop.updated_iteration == 5
    assert [e.trace_id for e in loop.evidence] == ["t0", "t1"] and [e.trace_id for e in loop.stale_evidence] == ["s0", "s1"]
    assert pruned.retired["loop-dup"].merged_into == "loop" and pruned.retired["stale-only"].reason == "all evidence retracted"
    assert pruned.retired["stale-only"].body == "# stale-only\nbody stale-only"  # nothing is deleted
    assert [e.pattern_id for e in pruned.index] == ["loop", "keep"] and pruned.log[-1].text.startswith("[prune] Merged")
    files = pruned.to_markdown_files()
    assert "wiki/retired.md" in files and "merged into loop" in files["wiki/retired.md"] and "wiki/retired.md" not in wiki.to_markdown_files()
    assert WikiState.from_document(pruned.to_document()).to_document() == pruned.to_document()
    # the original is untouched
    assert set(wiki.patterns) == {"loop", "loop-dup", "stale-only", "keep"}
    # hard rules
    with pytest.raises(WikiError, match="cited by an accepted skill"):
        wiki.apply_prune(PruneUpdate(retire=[{"name": "keep", "reason": "x"}]), iteration=5, bodies_shown=set())
    with pytest.raises(WikiError, match="cited by an accepted skill"):
        wiki.apply_prune(PruneUpdate(merges=[{"into": "loop", "from": ["keep"]}]), iteration=5, bodies_shown=set())
    with pytest.raises(WikiError, match="rewrites a body you were not shown"):
        wiki.apply_prune(PruneUpdate(merges=[{"into": "loop", "from": ["loop-dup"], "content": "new"}]), iteration=5, bodies_shown=set())
    with pytest.raises(WikiError, match="unknown pattern"):
        wiki.apply_prune(PruneUpdate(retire=[{"name": "ghost", "reason": "x"}]), iteration=5, bodies_shown=set())
    with pytest.raises(WikiError, match="twice"):
        wiki.apply_prune(PruneUpdate(merges=[{"into": "loop", "from": ["loop-dup"]}], retire=[{"name": "loop-dup", "reason": "x"}]), iteration=5, bodies_shown=set())
    with pytest.raises(WikiError, match="unexpected keys"):
        PruneUpdate.from_dict({"delete": []})
    # an empty update still leaves a log line and changes nothing else
    same = wiki.apply_prune(PruneUpdate(), iteration=6, bodies_shown=set())
    assert same.patterns.keys() == wiki.patterns.keys() and "nothing to change" in same.log[-1].text


def _traces():
    return [Trace(id=f"t{i}", outcome=TaskOutcome(task_id=f"task{i}", score=float(i >= 6), passed=i >= 6), text=f"body {i}") for i in range(10)]


def _tool(name, **args):
    return json.dumps({"tool": name, "args": args})


def _maintainer_reply(ids, iteration):
    if iteration == 1:
        return json.dumps({"create_patterns": [{"name": "p", "content": "# P\nx", "evidence": ids[:1]}, {"name": "p-dup", "content": "# P dup\nx", "evidence": ids[1:2]}],
                           "update_patterns": [], "update_index": "- [p](wiki/patterns/p.md): P; cause; fix\n- [p-dup](wiki/patterns/p-dup.md): same", "append_log": "log"})
    return json.dumps({"create_patterns": [], "update_patterns": [], "update_index": "- [p](wiki/patterns/p.md): P; cause; fix", "append_log": f"log {iteration}"})


class Recorder(Hooks):
    def __init__(self):
        self.stages, self.warnings = [], []

    def stage(self, name):
        self.stages.append(name)
        return super().stage(name)

    async def on_warning(self, message):
        self.warnings.append(message)


async def test_pruner_runs_when_due_and_the_maintainer_learns_what_was_retired():
    seen = {"pruner_input": None, "maintainer_after": None}

    def script(request):
        iteration = int(request.text.split("teration ")[1].split("\n")[0])
        if request.role == "wiki_maintainer":
            if iteration == 2:
                seen["maintainer_after"] = request.text
            ids = [line.split()[2] for line in request.text.splitlines() if line.startswith("=== Trace ")]
            return _maintainer_reply(ids, iteration)
        if request.role == "wiki_pruner":
            if iteration == 1:
                seen["pruner_input"] = request.text
                return json.dumps({"merges": [{"into": "p", "from": ["p-dup"], "reason": "same root cause"}], "retire": [], "append_log": "merged p-dup into p"})
            return json.dumps({"merges": [], "retire": [], "append_log": "nothing"})
        return _tool("finish", proposal={"action": "no_action", "rationale": "nothing"})

    revisions = MemoryRevisionStore()
    hooks = Recorder()
    report = await evolve(
        config=EvolveConfig(max_iterations=2, max_rejected_streak=5, prune_every=1),
        model=ScriptedChatModel(script),
        wiki_store=RevisionWikiStore(revisions, "ws"),
        skill_store=RevisionSkillStore(revisions, "ws"),
        trace_source=StaticTraceSource(_traces()),
        evaluator=type("E", (), {"evaluate": staticmethod(lambda *a, **k: None)})(),
        gate=None,
        baseline=Evaluation(ref=None, score=0.5),
        hooks=hooks,
    )
    assert [s for s in hooks.stages if s.startswith("prune:")] == ["prune:1", "prune:2"]
    assert set(report.wiki.patterns) == {"p"} and report.wiki.retired["p-dup"].merged_into == "p"
    assert "PROTECTED" not in seen["pruner_input"] and "- p-dup: " in seen["pruner_input"] and "## Pattern bodies shown (2 of 2)" in seen["pruner_input"]
    assert "## Retired patterns" in seen["maintainer_after"] and "p-dup: same root cause (merged into p)" in seen["maintainer_after"]
    # the budget trigger works without a cadence
    assert EvolveConfig(wiki_char_budget=10).wiki_char_budget == 10
    from skillwiki.harness import _prune_due
    assert _prune_due(EvolveConfig(wiki_char_budget=10), report.wiki, 3) and not _prune_due(EvolveConfig(wiki_char_budget=10_000_000), report.wiki, 3)
    assert not _prune_due(EvolveConfig(), report.wiki, 3)


def _proposer_script(proposal, *, calls):
    def script(request):
        calls.append(request.role)
        if request.role == "wiki_maintainer":
            ids = [line.split()[2] for line in request.text.splitlines() if line.startswith("=== Trace ")]
            return _maintainer_reply(ids, int(request.text.split("# Iteration ")[1].split("\n")[0]))
        turns = request.text.count("### Turn ")
        if turns < 4:
            return _tool("read_trace", id=f"t{turns}")
        return _tool("finish", proposal=proposal)
    return ScriptedChatModel(script)


class CrashingEvaluator:
    """Raises on the first candidate validation (a worker dying mid-validation), then scores normally."""

    def __init__(self, crash_on_candidate=True):
        self.crash, self.calls = crash_on_candidate, []

    async def evaluate(self, ref, skill_set, *, iteration, purpose):
        self.calls.append(purpose)
        if purpose == "candidate" and self.crash:
            self.crash = False
            raise RuntimeError("worker died")
        return Evaluation(ref=ref, score=0.5 if purpose == "baseline" else 0.9, per_task=[])


async def test_resume_finishes_the_interrupted_iteration_without_repeating_paid_role_stages():
    revisions = MemoryRevisionStore()
    stores = dict(wiki_store=RevisionWikiStore(revisions, "ws"), skill_store=RevisionSkillStore(revisions, "ws"),
                  checkpoint_store=RevisionCheckpointStore(revisions, "ws"), trace_source=StaticTraceSource(_traces()))
    proposal = {"action": "create", "name": "rule", "skill_md": SKILL, "purpose_md": "## Origin\nx", "citations": ["p"]}
    calls: list[str] = []
    evaluator = CrashingEvaluator()
    with pytest.raises(RuntimeError, match="worker died"):
        await evolve(config=EvolveConfig(max_iterations=2), model=_proposer_script(proposal, calls=calls), evaluator=evaluator, **stores)
    # W'_1 and the proposal were persisted before the crash
    _ref, wiki = await stores["wiki_store"].load()
    assert wiki.iteration == 1 and "p" in wiki.patterns and [e.iteration for e in wiki.impact] == [0]  # iteration 1's impact is recorded only on completion
    assert (await stores["checkpoint_store"].load(1))["proposal"]["name"] == "rule"
    first_calls = list(calls)
    hooks = Recorder()
    report = await evolve(config=EvolveConfig(max_iterations=2), model=_proposer_script(proposal, calls=calls), evaluator=evaluator, hooks=hooks, **stores)
    assert report.resumed_iteration == 1 and report.accepted == 1 and set(report.skills.skills) == {"rule"}
    assert report.iterations == 2 and [r.iteration for r in report.iteration_reports] == [1, 2]
    # iteration 1 re-ran validation only: no maintainer or proposer call before its validation stage
    resumed_stage_order = hooks.stages[: hooks.stages.index("validation:1") + 1]
    # iteration 2 re-proposes "rule", which now exists, so the proposer ends in no_action and nothing is validated
    assert resumed_stage_order == ["baseline", "validation:1"] and evaluator.calls == ["baseline", "candidate", "baseline", "candidate"]
    assert calls[len(first_calls):].count("wiki_maintainer") == 1  # only iteration 2's
    assert any("resumed iteration 1" in w for w in report.iteration_reports[0].warnings)
    assert report.wiki.impact[1].outcome == "accepted" and (await stores["checkpoint_store"].load(1)) == {"completed": True}
    # a third run starts fresh at iteration 3, nothing to resume
    again = await evolve(config=EvolveConfig(max_iterations=1), model=_proposer_script({"action": "no_action", "rationale": "n"}, calls=calls),
                         evaluator=evaluator, **stores)
    assert again.resumed_iteration is None and again.iteration_reports[0].iteration == 3


async def test_resume_reuses_a_recorded_validation_when_the_gate_crashed():
    revisions = MemoryRevisionStore()
    stores = dict(wiki_store=RevisionWikiStore(revisions, "ws"), skill_store=RevisionSkillStore(revisions, "ws"),
                  checkpoint_store=RevisionCheckpointStore(revisions, "ws"), trace_source=StaticTraceSource(_traces()))
    proposal = {"action": "create", "name": "rule", "skill_md": SKILL, "purpose_md": "## Origin\nx", "citations": ["p"]}

    class CrashingGate:
        def __init__(self):
            self.crash = True

        async def decide(self, best, candidate):
            if best is candidate:
                return __import__("skillwiki").Decision(accepted=False)
            if self.crash:
                self.crash = False
                raise RuntimeError("gate died")
            return __import__("skillwiki").Decision(accepted=candidate.score > best.score, feedback={"ok": True})

    gate = CrashingGate()
    evaluator = CrashingEvaluator(crash_on_candidate=False)
    with pytest.raises(RuntimeError, match="gate died"):
        await evolve(config=EvolveConfig(max_iterations=1), model=_proposer_script(proposal, calls=[]), evaluator=evaluator, gate=gate,
                     baseline=Evaluation(ref=None, score=0.5), **stores)
    assert "evaluation" in (await stores["checkpoint_store"].load(1))
    report = await evolve(config=EvolveConfig(max_iterations=1), model=_proposer_script(proposal, calls=[]), evaluator=evaluator, gate=gate,
                          baseline=Evaluation(ref=None, score=0.5), **stores)
    assert report.accepted == 1 and evaluator.calls == ["candidate"]  # the resumed run did not validate again
    assert any("reused the recorded validation" in w for w in report.iteration_reports[0].warnings)


async def test_duplicate_and_no_op_candidates_are_refused_before_validation():
    revisions = MemoryRevisionStore()
    stores = dict(wiki_store=RevisionWikiStore(revisions, "ws"), skill_store=RevisionSkillStore(revisions, "ws"), trace_source=StaticTraceSource(_traces()))
    rejected_skill = {"action": "create", "name": "rule", "skill_md": SKILL, "purpose_md": "## Origin\nx", "citations": ["p"]}
    observations = []

    def script(request):
        if request.role == "wiki_maintainer":
            ids = [line.split()[2] for line in request.text.splitlines() if line.startswith("=== Trace ")]
            return _maintainer_reply(ids, int(request.text.split("# Iteration ")[1].split("\n")[0]))
        iteration = int(request.text.split("# Iteration ")[1].split("\n")[0])
        turns = request.text.count("### Turn ")
        if turns < 4:
            return _tool("read_trace", id=f"t{turns}")
        if iteration == 1:
            return _tool("finish", proposal=rejected_skill)
        if turns == 4:  # iteration 2: re-propose the rejected skill -> refused inside the loop
            return _tool("finish", proposal=rejected_skill)
        observations.append(request.text.split("Observation:")[-1])
        if turns == 5:  # then a no-op patch -> refused too
            return _tool("finish", proposal={"action": "patch", "name": "other", "edits": [{"op": "replace", "target": "other thing", "content": "other thing"}], "rationale": "noop"})
        return _tool("finish", proposal={"action": "no_action", "rationale": "gave up"})

    class Scorer:
        def __init__(self):
            self.calls = 0

        async def evaluate(self, ref, skill_set, *, iteration, purpose):
            self.calls += 1
            return Evaluation(ref=ref, score=0.5 if purpose == "baseline" else 0.4)  # every candidate is worse

    scorer = Scorer()
    hooks = Recorder()
    report = await evolve(
        config=EvolveConfig(max_iterations=2, max_rejected_streak=5),
        model=ScriptedChatModel(script), evaluator=scorer, hooks=hooks,
        initial_skills=[SkillDocument.parse_skill_md(SKILL_B)], **stores,
    )
    assert report.rejected == 1 and report.iteration_reports[1].outcome == "no_action" and scorer.calls == 2  # baseline + one validation
    assert "identical to the one REJECTED at iteration 1" in observations[0] and "changes nothing" in observations[1]
    # the harness backstop covers a candidate the proposer did not know was judged
    from skillwiki.harness import _known_candidate
    digest = SkillSet.from_documents([SkillDocument.parse_skill_md(SKILL_B), SkillDocument.parse_skill_md(SKILL)]).content_digest()
    known = _known_candidate(report.wiki, digest, report.skills)
    assert known["reason"] == "duplicate_candidate" and known["duplicate_of_iteration"] == 1
    assert _known_candidate(report.wiki, report.skills.content_digest(), report.skills)["reason"] == "no_change"
    assert _known_candidate(report.wiki, "nope", report.skills) is None


async def test_completion_checkpoints_are_written_only_for_iterations_that_recorded_a_proposal():
    revisions = MemoryRevisionStore()
    checkpoints = RevisionCheckpointStore(revisions, "ws")
    await evolve(
        config=EvolveConfig(max_iterations=2, max_rejected_streak=5),
        model=_proposer_script({"action": "no_action", "rationale": "n"}, calls=[]),
        wiki_store=RevisionWikiStore(revisions, "ws"), skill_store=RevisionSkillStore(revisions, "ws"), checkpoint_store=checkpoints,
        trace_source=StaticTraceSource(_traces()), evaluator=CrashingEvaluator(crash_on_candidate=False),
    )
    assert await revisions.list("ws", "checkpoint") == []  # two no_action iterations, no checkpoint chain growth
    # a resumed run re-materialises the MODULE's candidate, not the host's
    seen = []

    class RecordingStore(RevisionSkillStore):
        async def propose(self, current_ref, current, proposal, candidate, *, iteration, wiki_ref):
            seen.append(candidate.skills["rule"].body)
            host = candidate.copy()
            host.skills["rule"].body = candidate.skills["rule"].body + "\n<!-- host -->"
            return await super().propose(current_ref, current, proposal, host, iteration=iteration, wiki_ref=wiki_ref)

    revisions2 = MemoryRevisionStore()
    stores = dict(wiki_store=RevisionWikiStore(revisions2, "ws"), skill_store=RecordingStore(revisions2, "ws"),
                  checkpoint_store=RevisionCheckpointStore(revisions2, "ws"), trace_source=StaticTraceSource(_traces()))
    proposal = {"action": "create", "name": "rule", "skill_md": SKILL, "purpose_md": "## Origin\nx", "citations": ["p"]}

    class CrashingGate:
        crash = True

        async def decide(self, best, candidate):
            if best is candidate:
                return __import__("skillwiki").Decision(accepted=False)
            if self.crash:
                self.crash = False
                raise RuntimeError("gate died")
            return __import__("skillwiki").Decision(accepted=True)

    gate = CrashingGate()
    with pytest.raises(RuntimeError):
        await evolve(config=EvolveConfig(max_iterations=1), model=_proposer_script(proposal, calls=[]), evaluator=CrashingEvaluator(False), gate=gate,
                     baseline=Evaluation(ref=None, score=0.5), **stores)
    report = await evolve(config=EvolveConfig(max_iterations=1), model=_proposer_script(proposal, calls=[]), evaluator=CrashingEvaluator(False), gate=gate,
                          baseline=Evaluation(ref=None, score=0.5), **stores)
    assert seen == ["# Rule\nDo the thing.\n"] * 2  # both materialisations started from the module's candidate
    assert report.skills.skills["rule"].body.count("<!-- host -->") == 1
