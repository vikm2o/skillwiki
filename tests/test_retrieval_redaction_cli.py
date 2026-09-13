# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
import asyncio
import json

from skillwiki import (
    EMAIL,
    EvolveConfig,
    MemoryRevisionStore,
    RevisionSkillStore,
    RevisionTraceStore,
    RevisionWikiStore,
    ScriptedChatModel,
    StaticTraceSource,
    TaskOutcome,
    Trace,
    evolve,
    redact_trace,
    regex_redactor,
    summarize_skill_usage,
)
from skillwiki.cli import main
from skillwiki.gates import Evaluation
from skillwiki.roles.proposer import SkillProposer
from skillwiki.skills import Proposal, SkillDocument, SkillSet
from skillwiki.stores.file import FileRevisionStore
from skillwiki.wiki import WikiState


def _trace(i, passed, skills=()):
    return Trace(id=f"t{i}", outcome=TaskOutcome(f"task{i}", 1.0 if passed else 0.0, passed), text=f"body {i}", skills_in_play=list(skills))


def test_skills_in_play_round_trip_render_and_usage_table():
    traces = [_trace(1, False, ["rule", "other"]), _trace(2, False, ["rule"]), _trace(3, True, ["other"]), _trace(4, True)]
    assert Trace.from_document(traces[0].to_document()).skills_in_play == ["rule", "other"]
    assert "skills in play: rule, other" in traces[0].rendered(char_cap=100)
    table = summarize_skill_usage(traces)
    assert table.splitlines()[2:] == ["| rule | 2 | 0 |", "| other | 1 | 1 |"]  # most-failing first
    assert summarize_skill_usage([_trace(5, True)]) == ""  # no skills named: no table, nothing to fold


async def test_proposer_sees_usage_in_the_opening_and_in_read_skill():
    traces = [_trace(1, False, ["rule"]), _trace(2, True, ["rule"]), _trace(3, False)]
    skills = SkillSet.from_documents([SkillDocument(name="rule", body="# Rule\nDo the thing.\n", purpose="p", frontmatter={"description": "d"})])
    model = ScriptedChatModel([
        json.dumps({"tool": "read_skill", "args": {"name": "rule"}}),
        json.dumps({"tool": "finish", "args": {"proposal": {"action": "no_action", "rationale": "fine"}}}),
    ])
    proposal, pad = await SkillProposer(model, EvolveConfig(react_min_trace_reads=0)).propose(WikiState(), skills, traces, iteration=1)
    assert proposal.action == "no_action"
    opening = model.calls[0].text
    assert "## Skill usage across the training traces" in opening and "| rule | 1 | 1 |" in opening
    assert "in play in 2 training trace(s): 1 failing, 1 passing" in pad.turns[0].observation


class _Evaluator:
    async def evaluate(self, ref, skills, *, iteration, purpose):
        return Evaluation(ref=ref, score=0.5)


async def test_redaction_applies_before_the_roles_and_the_trace_store(tmp_path):
    redactor = regex_redactor({"email": EMAIL, "ticket": r"TCK-\d+"})
    trace = Trace(id="t1", outcome=TaskOutcome("k", 0.0, False), text="reviewer ann@example.com filed TCK-42")
    copy = redact_trace(trace, redactor)
    assert copy.text == "reviewer [REDACTED:email] filed [REDACTED:ticket]" and trace.text.startswith("reviewer ann@")
    assert redact_trace(copy, redactor) is copy  # nothing to do: same object back

    revisions = MemoryRevisionStore()
    model = ScriptedChatModel([
        json.dumps({"create_patterns": [], "update_patterns": [], "update_index": "", "append_log": "looked"}),
        json.dumps({"tool": "finish", "args": {"proposal": {"action": "no_action", "rationale": "n"}}}),
    ])
    await evolve(
        config=EvolveConfig(max_iterations=1, max_rejected_streak=5), model=model, trace_source=StaticTraceSource([trace]),
        wiki_store=RevisionWikiStore(revisions, "ws"), skill_store=RevisionSkillStore(revisions, "ws"), trace_store=RevisionTraceStore(revisions, "ws"),
        evaluator=_Evaluator(), redact=redactor,
    )
    stored = await RevisionTraceStore(revisions, "ws").get("t1")
    assert "[REDACTED:email]" in stored.text and "ann@" not in stored.text
    seen_by_model = "".join(r.text for r in model.calls)
    assert "ann@example.com" not in seen_by_model and "[REDACTED:email]" in seen_by_model


async def _seed_two_skill_revisions(root):
    store = FileRevisionStore(root)
    skills = RevisionSkillStore(store, "ws")
    first = SkillSet.from_documents([SkillDocument(name="rule", body="# Rule\nDo the thing.\n", purpose="p", frontmatter={"description": "d"})])
    ref = await skills.seed(first)
    second = first.copy()
    second.skills["rule"].body = "# Rule\nDo the thing carefully.\n"
    candidate = await skills.propose(ref, first, Proposal(action="patch", name="rule", rationale="r"), second, iteration=1, wiki_ref=None)
    await skills.accept(candidate, expected_ref=ref)
    await RevisionWikiStore(store, "ws").save(WikiState(), expected_ref=None, iteration=0)


def test_cli_show_export_impact_history_diff_over_a_file_store(tmp_path, capsys):
    root = tmp_path / "store"
    asyncio.run(_seed_two_skill_revisions(root))  # the CLI owns its own event loop, so the test must not run inside one

    assert main(["--store", f"file:{root}", "--workspace", "ws", "show"]) == 0
    out = capsys.readouterr().out
    assert "1 skill(s)" in out and "- rule: d" in out
    assert main(["--store", f"file:{root}", "--workspace", "ws", "export", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "skills" / "rule" / "SKILL.md").read_text().endswith("carefully.\n")
    assert main(["--store", f"file:{root}", "--workspace", "ws", "impact"]) == 0
    assert main(["--store", f"file:{root}", "--workspace", "ws", "history", "--kind", "skills"]) == 0
    assert capsys.readouterr().out.count("skills ") == 2
    assert main(["--store", f"file:{root}", "--workspace", "ws", "diff"]) == 0
    diff = capsys.readouterr().out
    assert "-Do the thing." in diff and "+Do the thing carefully." in diff
    assert main(["--store", "s3:bucket", "show"]) == 2
    assert "unsupported store" in capsys.readouterr().err
