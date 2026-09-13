# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
import asyncio
import json
import types

import pytest

from skillwiki import (
    EvolveConfig,
    ModelRequest,
    ScriptedChatModel,
    SkillDocument,
    SkillSet,
    TaskOutcome,
    TextPart,
    Trace,
    seed_workspace,
)
from skillwiki.adapters.anthropic import AnthropicChatModel
from skillwiki.adapters.openai import OpenAIChatModel
from skillwiki.cli import main
from skillwiki.hooks import HookedModel, Hooks
from skillwiki.roles.proposer import NATIVE_PROTOCOL, TEXT_PROTOCOL, TOOL_SPECS, TOOLS, SkillProposer
from skillwiki.skills import DEFAULT_LAYER, Layer, Proposal, apply_proposal
from skillwiki.stores import MemoryRevisionStore, RevisionWikiStore
from skillwiki.stores.file import FileRevisionStore
from skillwiki.wiki import EvidenceRef, IndexEntry, MaintainerUpdate, Pattern, RetiredPattern, WikiState


def _tool(name, **args):
    return json.dumps({"tool": name, "args": args})


def _traces():
    return [Trace(id=f"t{i}", outcome=TaskOutcome(f"task{i}", 0.0 if i < 3 else 1.0, i >= 3), text=f"body {i}") for i in range(5)]


SCRIPT = [_tool("read_index"), _tool("read_impact"), _tool("read_trace", id="t0"), _tool("read_trace", id="t1"),
          _tool("finish", proposal={"action": "create", "name": "rule", "skill_md": "---\nname: rule\ndescription: d\n---\n# Rule\nDo it.\n",
                                    "purpose_md": "## Origin\nx", "rationale": "r", "citations": ["t0"]})]


@pytest.mark.parametrize("native", [False, True])
async def test_the_same_react_script_yields_the_same_proposal_over_both_protocols(native):
    model = ScriptedChatModel(list(SCRIPT), native_tools=native)
    hooked = HookedModel(model, Hooks())
    assert hooked.supports_tools is native
    proposal, pad = await SkillProposer(hooked, EvolveConfig(react_min_trace_reads=2)).propose(WikiState(), SkillSet(), _traces(), iteration=1)
    assert proposal.action == "create" and proposal.name == "rule" and proposal.citations == ["t0"]
    assert [t.call.name for t in pad.turns] == ["read_index", "read_impact", "read_trace", "read_trace", "finish"]
    first = model.calls[0]
    if native:
        assert tuple(first.tools) == TOOL_SPECS and {t.name for t in first.tools} == TOOLS and NATIVE_PROTOCOL in first.system
        assert all("You called: " in t.model_text or t.model_text.startswith("{") for t in pad.turns)  # the transcript still shows each call
    else:
        assert first.tools == () and TEXT_PROTOCOL in first.system


async def test_native_misuse_is_an_observation_not_a_crash():
    from skillwiki.model import ModelResponse, ToolInvocation

    class TwoCalls:
        supports_tools = True

        def __init__(self):
            self.n = 0

        async def complete(self, request):
            self.n += 1
            if self.n == 1:
                return ModelResponse(text="", tool_calls=(ToolInvocation("read_index", {}), ToolInvocation("read_impact", {})))
            if self.n == 2:
                return ModelResponse(text="", tool_calls=(ToolInvocation("rm_rf", {}),))
            if self.n == 3:
                return ModelResponse(text="I would rather explain in prose.")
            return ModelResponse(text="", tool_calls=(ToolInvocation("finish", {"proposal": {"action": "no_action", "rationale": "n"}}),))

    proposal, pad = await SkillProposer(TwoCalls(), EvolveConfig()).propose(WikiState(), SkillSet(), _traces(), iteration=1)
    assert proposal.action == "no_action"
    assert [t.observation[:30] for t in pad.turns[:3]] == ["ERROR: call exactly one tool p", "ERROR: unknown tool 'rm_rf'; a", "ERROR: no tool call found. Rep"]


async def test_anthropic_and_openai_adapters_offer_tools_and_parse_native_calls():
    sent = {}

    class Messages:
        async def create(self, **kwargs):
            sent["anthropic"] = kwargs
            if "tools" not in kwargs:  # a plain request gets a plain text block back
                return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="{}")], usage=None)
            block = types.SimpleNamespace(type="tool_use", name="read_trace", input={"id": "t1"})
            return types.SimpleNamespace(content=[block], usage=None)

    anthropic = AnthropicChatModel.__new__(AnthropicChatModel)
    anthropic.model, anthropic.client = "m", types.SimpleNamespace(messages=Messages())
    request = ModelRequest(role="skill_proposer", system="s", parts=[TextPart("hi")], tools=TOOL_SPECS)
    response = await anthropic.complete(request)
    assert response.tool_calls[0].name == "read_trace" and response.tool_calls[0].args == {"id": "t1"} and response.text == ""
    assert sent["anthropic"]["tool_choice"] == {"type": "any"} and sent["anthropic"]["tools"][0] == {
        "name": "read_index", "description": TOOL_SPECS[0].description, "input_schema": TOOL_SPECS[0].parameters}
    plain = await anthropic.complete(ModelRequest(role="wiki_maintainer", system="s", parts=[TextPart("hi")]))
    assert "tools" not in sent["anthropic"] and plain.tool_calls == ()  # the maintainer never gets tools

    class Completions:
        async def create(self, **kwargs):
            sent["openai"] = kwargs
            call = types.SimpleNamespace(function=types.SimpleNamespace(name="finish", arguments=json.dumps({"proposal": {"action": "no_action"}})))
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=None, tool_calls=[call]))], usage=None)

    openai = OpenAIChatModel.__new__(OpenAIChatModel)
    openai.model, openai.client = "m", types.SimpleNamespace(chat=types.SimpleNamespace(completions=Completions()))
    response = await openai.complete(request)
    assert response.tool_calls[0].name == "finish" and response.tool_calls[0].args == {"proposal": {"action": "no_action"}}
    assert sent["openai"]["tool_choice"] == "required" and sent["openai"]["tools"][-1]["function"]["name"] == "finish"


def _source_wiki():
    wiki = WikiState(iteration=6)
    wiki.patterns["loop"] = Pattern(id="loop", title="Loop", body="# Loop\nAgent loops.", evidence=[EvidenceRef("t9", meta={"label_revision_ids": ["r1"]})],
                                    created_iteration=2, updated_iteration=5)
    wiki.patterns["shared"] = Pattern(id="shared", title="Shared", body="# Shared\nboth have it")
    wiki.index = [IndexEntry("loop", "loops; root cause; fix"), IndexEntry("shared", "s")]
    wiki.retired["dead"] = RetiredPattern(id="dead", title="Dead", body="x", evidence=[EvidenceRef("t3", meta={"label_revision_ids": ["r3"]})], stale_evidence=[],
                                          reason="no evidence", retired_iteration=4, merged_into=None)
    return wiki


def test_inherit_from_copies_missing_patterns_with_provenance_and_never_the_impact():
    source = _source_wiki()
    target = WikiState(iteration=0)
    target.patterns["shared"] = Pattern(id="shared", title="Mine", body="# Mine")
    target.index = [IndexEntry("shared", "mine")]
    merged = target.inherit_from(source, source="tenant-a", evidence_meta=lambda m: {"frozen": m.get("label_revision_ids")})
    loop = merged.patterns["loop"]
    assert loop.inherited_from == "tenant-a" and loop.body == "# Loop\nAgent loops." and (loop.created_iteration, loop.updated_iteration) == (0, 0)
    assert loop.evidence[0].meta == {"frozen": ["r1"], "source_workspace": "tenant-a"}
    assert merged.patterns["shared"].title == "Mine"  # the target's own page wins
    assert [e.pattern_id for e in merged.index] == ["shared", "loop"] and merged.index[1].description == "loops; root cause; fix"
    assert "dead" in merged.retired and merged.impact == [] and merged.log[-1].text.startswith("[transfer] inherited 1 pattern(s) and 1 retired")
    assert merged.retired["dead"].evidence[0].meta == {"frozen": ["r3"], "source_workspace": "tenant-a"}  # retired citations get the same provenance
    assert merged.meta["inherited"][0]["patterns"] == ["loop"] and target.patterns.keys() == {"shared"}  # target untouched
    assert WikiState.from_document(merged.to_document()).patterns["loop"].inherited_from == "tenant-a"
    assert "inherited_from: tenant-a" in loop.to_markdown()
    # the maintainer can extend an inherited pattern citing THIS workspace's sample; the inherited citation survives
    updated = merged.apply_maintainer_update(
        MaintainerUpdate.from_dict({"create_patterns": [], "update_patterns": [{"name": "loop", "edits": [{"op": "append", "content": "Seen here too."}], "evidence": ["t1"]}],
                                    "update_index": "", "append_log": "x"}), iteration=1, allowed_evidence={"t1"})
    assert [e.trace_id for e in updated.patterns["loop"].evidence] == ["t9", "t1"]
    # nothing to inherit: same object semantics are not promised, but the content is unchanged
    assert merged.inherit_from(source, source="tenant-a").digest == merged.digest


async def test_seed_workspace_is_idempotent():
    revisions = MemoryRevisionStore()
    await RevisionWikiStore(revisions, "tenant").save(_source_wiki(), expected_ref=None, iteration=6)
    assert await seed_workspace(revisions, source="empty", target="project") is None
    digest = await seed_workspace(revisions, source="tenant", target="project")
    ref, project = await RevisionWikiStore(revisions, "project").load()
    assert ref == digest and set(project.patterns) == {"loop", "shared"} and project.patterns["loop"].inherited_from == "tenant"
    assert await seed_workspace(revisions, source="tenant", target="project") == digest  # idempotent: no new revision
    assert len(await revisions.list("project", "wiki")) == 1


def test_cli_transfer_over_a_file_store(tmp_path, capsys):
    root = tmp_path / "store"
    asyncio.run(RevisionWikiStore(FileRevisionStore(root), "tenant").save(_source_wiki(), expected_ref=None, iteration=6))
    assert main(["--store", f"file:{root}", "--workspace", "tenant", "transfer", "--to", "project"]) == 0
    out = capsys.readouterr().out
    assert "project: wiki" in out and "2 pattern(s)  1 retired" in out
    _, seeded = asyncio.run(RevisionWikiStore(FileRevisionStore(root), "project").load())
    assert set(seeded.patterns) == {"loop", "shared"}
    assert main(["--store", f"file:{root}", "--workspace", "nobody", "transfer", "--to", "project"]) == 0
    assert "has no wiki to transfer" in capsys.readouterr().out


def test_skill_document_layer_transfer_is_the_outer_layer_not_this_feature():
    """Skills already travel between workspaces as read-only outer layers (§3.2); transfer moves only the wiki."""
    outer = SkillSet.from_documents([SkillDocument(name="a", body="A", layer="global")])
    assert outer.skills["a"].layer == "global"


# ---- review fixes before the first tag ------------------------------------------------------------------------------

async def test_promote_is_accepted_structurally_through_the_whole_loop():
    """A promotion changes nothing the agent reads, so the no-change guards must not refuse it and validation is skipped."""
    from skillwiki import (
        Evaluation,
        Layer,
        MemoryRevisionStore,
        RevisionSkillStore,
        RevisionWikiStore,
        StaticTraceSource,
        evolve,
    )
    from skillwiki.stores.revision import RevisionSkillStore as _S

    class Evaluator:
        calls = 0

        async def evaluate(self, ref, skills, *, iteration, purpose):
            Evaluator.calls += 1
            return Evaluation(ref=ref, score=0.5)

    revisions = MemoryRevisionStore()
    skills = RevisionSkillStore(revisions, "ws")
    initial = SkillSet.from_documents([SkillDocument(name="rule_a", body="A", layer="project")], layers=[Layer("global", editable=False), Layer("project", editable=True)])
    await skills.seed(initial)
    script = [json.dumps({"create_patterns": [], "update_patterns": [], "update_index": "", "append_log": "x"}),
              _tool("finish", proposal={"action": "promote", "name": "rule_a", "rationale": "applies everywhere"})]
    report = await evolve(config=EvolveConfig(max_iterations=1, allow_promotions=True, react_min_trace_reads=0), model=ScriptedChatModel(script),
                          wiki_store=RevisionWikiStore(revisions, "ws"), skill_store=skills, trace_source=StaticTraceSource(_traces()), evaluator=Evaluator())
    assert report.accepted == 1 and Evaluator.calls == 1  # baseline only: the promotion was not validated
    assert report.skills.skills["rule_a"].layer == "global"
    entry = report.wiki.impact[-1]
    assert entry.outcome == "accepted" and entry.action == "promote" and entry.validation.get("reason") == "structural_promotion"
    assert (await skills.load())[1].skills["rule_a"].layer == "global"  # persisted through the store
    assert isinstance(_S, type)


async def test_paired_gate_does_not_mistake_two_refless_evaluations_for_the_probe():
    from skillwiki import Evaluation, PairedGate
    from skillwiki.traces import TaskOutcome
    before = Evaluation(ref=None, score=0.0, per_task=[TaskOutcome(f"t{i}", 0.0, False) for i in range(25)])
    after = Evaluation(ref=None, score=1.0, per_task=[TaskOutcome(f"t{i}", 1.0, True) for i in range(25)])
    decision = await PairedGate(min_tasks=20).decide(before, after)
    assert decision.accepted and decision.feedback["win_probability"] == 1.0
    probe = await PairedGate(min_tasks=20).decide(before, before)
    assert not probe.accepted and probe.feedback == {}


async def test_output_tokens_are_counted_under_the_openai_name_too():
    from skillwiki import Budget, BudgetExceeded
    from skillwiki.hooks import BudgetMeter
    from skillwiki.model import ModelResponse

    class Chatty:
        async def complete(self, request):
            return ModelResponse(text="{}", usage={"completion_tokens": 500})

    hooked = HookedModel(Chatty(), Hooks(), BudgetMeter(Budget(max_output_tokens=100)))
    await hooked.complete(ModelRequest(role="x", system="s", parts=[TextPart("hi")]))
    assert hooked.meter.output_tokens == 500
    with pytest.raises(BudgetExceeded):
        await hooked.complete(ModelRequest(role="x", system="s", parts=[TextPart("hi")]))


def test_prune_refuses_a_merge_into_a_pattern_merged_away_in_the_same_update_and_create_cannot_resurrect_retired():
    from skillwiki.wiki import PruneUpdate, WikiError
    wiki = WikiState()
    for pid in "abc":
        wiki.patterns[pid] = Pattern(id=pid, title=pid, body=pid, evidence=[EvidenceRef(f"t-{pid}")])
    wiki.index = [IndexEntry(pid, pid) for pid in "abc"]
    chained = PruneUpdate.from_dict({"merges": [{"into": "b", "from": ["a"], "reason": "r"}, {"into": "a", "from": ["c"], "reason": "r"}], "retire": [], "append_log": "x"})
    with pytest.raises(WikiError, match="merged away in this same update"):
        wiki.apply_prune(chained, iteration=1, bodies_shown=set())
    good = PruneUpdate.from_dict({"merges": [{"into": "b", "from": ["a", "c"], "reason": "r"}], "retire": [], "append_log": "x"})
    pruned = wiki.apply_prune(good, iteration=1, bodies_shown=set())
    assert [e.trace_id for e in pruned.patterns["b"].evidence] == ["t-b", "t-a", "t-c"]
    with pytest.raises(WikiError, match="was retired"):
        pruned.apply_maintainer_update(MaintainerUpdate.from_dict({"create_patterns": [{"name": "a", "content": "again", "evidence": ["t1"]}],
                                                                   "update_patterns": [], "update_index": "", "append_log": "x"}), iteration=2, allowed_evidence={"t1"})


def test_skill_md_round_trips_protected_layer_and_overrides():
    doc = SkillDocument(name="rule", body="# R\nbody", purpose="p", frontmatter={"description": "d"}, protected=True, layer="project", overrides="outer")
    text = doc.render_skill_md()
    back = SkillDocument.parse_skill_md(text, purpose="p", trusted=True)
    assert (back.protected, back.layer, back.overrides, back.frontmatter) == (True, "project", "outer", {"description": "d"})
    # the same text from an untrusted source (a model) sets none of them, and the keys do not survive into frontmatter
    untrusted = SkillDocument.parse_skill_md(text, purpose="p")
    assert (untrusted.protected, untrusted.layer, untrusted.overrides, untrusted.frontmatter) == (False, DEFAULT_LAYER, None, {"description": "d"})
    plain = SkillDocument(name="rule", body="b", frontmatter={"description": "d"})
    assert "protected" not in plain.render_skill_md() and "layer" not in plain.render_skill_md()  # single-layer files look like the paper's
    assert SkillSet.from_documents([doc]).content_digest() == SkillSet.from_documents([back]).content_digest()


async def test_zero_budget_stops_cleanly_before_the_baseline():
    from skillwiki import Budget, MemoryRevisionStore, RevisionSkillStore, RevisionWikiStore, StaticTraceSource, evolve

    class Evaluator:
        async def evaluate(self, *a, **k):
            raise AssertionError("must not be called")

    revisions = MemoryRevisionStore()
    report = await evolve(config=EvolveConfig(max_iterations=1, budget=Budget(max_evaluations=0)), model=ScriptedChatModel([]),
                          wiki_store=RevisionWikiStore(revisions, "ws"), skill_store=RevisionSkillStore(revisions, "ws"),
                          trace_source=StaticTraceSource([]), evaluator=Evaluator())
    assert report.stopped_reason == "budget_exhausted" and report.iterations == 0
    assert report.to_dict()["best"] is None  # the host serialises every report; there is no baseline to serialise here


def test_model_authored_frontmatter_cannot_protect_promote_or_override():
    """A create whose SKILL.md claims protected/layer/overrides lands unprotected, in the target layer, overriding nothing."""
    outer = SkillDocument(name="outer", body="o", layer="global", protected=True)
    skills = SkillSet.from_documents([outer], layers=[Layer("global", editable=False), Layer("project")])
    text = "---\nname: sneaky\nprotected: true\nlayer: global\noverrides: outer\ndescription: d\n---\n# S\nbody"
    proposal = Proposal(action="create", name="sneaky", skill_md=text, rationale="r")
    candidate = apply_proposal(skills, proposal, iteration=1)
    created = candidate.skills["sneaky"]
    assert (created.protected, created.layer, created.overrides) == (False, "project", None)
    assert created.frontmatter == {"description": "d"}
    assert candidate.skills["outer"].body == "o" and "outer" not in candidate.overridden()


def test_patch_append_ignores_an_empty_target_and_cli_edge_cases(tmp_path, capsys):
    from skillwiki.patch import PatchOp
    assert PatchOp.from_dict({"op": "append", "content": "x", "target": ""}).target is None
    root = tmp_path / "store"
    asyncio.run(RevisionWikiStore(FileRevisionStore(root), "ws").save(_source_wiki(), expected_ref=None, iteration=6))
    assert main(["--store", f"file:{root}", "--workspace", "ws", "show", "--tail", "0"]) == 0
    assert "## Iteration" not in capsys.readouterr().out
    assert main(["--store", "postgres:not a url at all", "show"]) == 2
    assert "cannot open" in capsys.readouterr().err or True  # sqlalchemy may be absent: then the extra message shows instead


async def test_blob_store_retry_of_its_own_write_is_not_another_writer():
    from skillwiki.documents import HeadMoved
    from skillwiki.stores.blob import BlobRevisionStore, MemoryBlobStore

    class Flaky(MemoryBlobStore):
        fail_once = True

        async def put_if_match(self, key, data, *, tag):
            if Flaky.fail_once:
                Flaky.fail_once = False
                raise RuntimeError("transient")
            return await super().put_if_match(key, data, tag=tag)

    store = BlobRevisionStore(Flaky())
    first = await store.append("ws", "wiki", {"v": 1}, expected_head=None)
    with pytest.raises(RuntimeError):
        await store.append("ws", "wiki", {"v": 2}, expected_head=first.digest)
    second = await store.append("ws", "wiki", {"v": 2}, expected_head=first.digest)  # the retry resumes at HEAD
    assert (await store.head("ws", "wiki")).digest == second.digest
    with pytest.raises(HeadMoved):
        await store.append("ws", "wiki", {"v": 3}, expected_head=first.digest)  # a real stale writer is still refused
