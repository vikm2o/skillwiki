# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Skill layers (override / promote / read-only guards), content digests, schema compatibility and PairedGate."""

import pytest

from skillwiki import (
    Evaluation,
    PairedGate,
    Proposal,
    ProposalError,
    SkillDocument,
    SkillSet,
    TaskOutcome,
    apply_proposal,
    unified_diff,
)
from skillwiki.skills import Layer

GLOBAL = SkillDocument(name="brand_tag", body="Check the brand tag is visible.", frontmatter={"description": "tag"}, layer="global")
TENANT = SkillDocument(name="hemline", body="Hemline must be straight.", frontmatter={"description": "hem"}, layer="tenant")
LOCAL = SkillDocument(name="sleeve", body="Sleeves symmetric.", frontmatter={"description": "sleeve"}, layer="project")
LAYERS = [Layer("global", False), Layer("tenant", False), Layer("project", True)]
SKILL_MD = "---\nname: brand_tag_lenient\ndescription: lenient tag\n---\n# Tag\nTag may be folded for this project.\n"


def _set():
    return SkillSet.from_documents([GLOBAL, TENANT, LOCAL], layers=LAYERS)


def test_read_only_layers_refuse_patch_and_retire_but_allow_override():
    skills = _set()
    with pytest.raises(ProposalError, match="read-only layer 'global'.*override"):
        apply_proposal(skills, Proposal.from_dict({"action": "patch", "name": "brand_tag", "edits": [{"op": "append", "content": "x"}]}), iteration=1)
    with pytest.raises(ProposalError, match="read-only"):
        apply_proposal(skills, Proposal.from_dict({"action": "retire", "name": "hemline"}), iteration=1)
    proposal = Proposal.from_dict({"action": "override", "name": "brand_tag_lenient", "overrides": "brand_tag", "skill_md": SKILL_MD, "purpose_md": "## Origin\nfolded tags are fine here"})
    candidate = apply_proposal(skills, proposal, iteration=2)
    new = candidate.skills["brand_tag_lenient"]
    assert new.layer == "project" and new.overrides == "brand_tag" and "Overrides brand_tag (global)" in new.purpose
    # the outer skill stays in the set but disappears from what the agent reads
    assert "brand_tag" in candidate.skills and [s.name for s in candidate.effective()] == ["brand_tag_lenient", "hemline", "sleeve"]
    assert "(overridden by brand_tag_lenient)" in candidate.render_for_prompt() and "## Layer: global (read-only)" in candidate.render_for_prompt()
    # an overridden skill can neither be overridden again nor edited through the back door
    with pytest.raises(ProposalError, match="already overridden"):
        apply_proposal(candidate, Proposal.from_dict({"action": "override", "name": "other", "overrides": "brand_tag", "skill_md": SKILL_MD.replace("brand_tag_lenient", "other"), "purpose_md": "x"}), iteration=3)
    # creating lands in the innermost editable layer; editable skills cannot be "overridden", only patched
    created = apply_proposal(skills, Proposal.from_dict({"action": "create", "name": "cuffs", "skill_md": SKILL_MD.replace("brand_tag_lenient", "cuffs"), "purpose_md": "x"}), iteration=1)
    assert created.skills["cuffs"].layer == "project"
    with pytest.raises(ProposalError, match="is editable; use action 'patch'"):
        apply_proposal(skills, Proposal.from_dict({"action": "override", "name": "s2", "overrides": "sleeve", "skill_md": SKILL_MD.replace("brand_tag_lenient", "s2"), "purpose_md": "x"}), iteration=1)


def test_protected_beats_override_and_promote_is_gated():
    protected = SkillDocument(name="safety", body="Never pass an asset with a missing garment.", layer="global", protected=True)
    skills = SkillSet.from_documents([protected, LOCAL], layers=[Layer("global", False), Layer("project", True)])
    with pytest.raises(ProposalError, match="protected"):
        apply_proposal(skills, Proposal.from_dict({"action": "override", "name": "s", "overrides": "safety", "skill_md": SKILL_MD.replace("brand_tag_lenient", "s"), "purpose_md": "x"}), iteration=1)
    with pytest.raises(ProposalError, match="unknown action 'promote'"):
        Proposal.from_dict({"action": "promote", "name": "sleeve"})
    promoted = apply_proposal(skills, Proposal.from_dict({"action": "promote", "name": "sleeve", "rationale": "applies everywhere"}, allow_promotions=True), iteration=4)
    assert promoted.skills["sleeve"].layer == "global" and "Promoted to layer global" in promoted.skills["sleeve"].purpose
    with pytest.raises(ProposalError, match="read-only layer .global."):  # a promoted skill is now owned by the outer layer
        apply_proposal(promoted, Proposal.from_dict({"action": "promote", "name": "sleeve"}, allow_promotions=True), iteration=5)


def test_single_layer_sets_render_like_the_paper_and_documents_round_trip():
    flat = SkillSet.from_documents([SkillDocument(name="a", body="A"), SkillDocument(name="b", body="B")])
    assert set(flat.to_files()) == {"skills/a/SKILL.md", "skills/a/PURPOSE.md", "skills/b/SKILL.md", "skills/b/PURPOSE.md"}
    assert "## Layer" not in flat.render_for_prompt()
    layered = _set()
    assert "skills/global/brand_tag/SKILL.md" in layered.to_files() and "skills/project/sleeve/PURPOSE.md" in layered.to_files()
    assert "+++ skills/project/sleeve/SKILL.md" in unified_diff(SkillSet(layers=LAYERS), layered)
    doc = layered.to_document()
    assert doc["schema"] == 2 and [layer["name"] for layer in doc["layers"]] == ["global", "tenant", "project"]
    assert SkillSet.from_document(doc).to_document() == doc
    # a 0.1 document (no schema, no layers) loads as one editable layer
    legacy = {"version": 1, "skills": [{"name": "a", "body": "A", "purpose": "", "frontmatter": {}, "protected": False}]}
    loaded = SkillSet.from_document(legacy)
    assert loaded.layers == [Layer("local", True)] and loaded.skills["a"].layer == "local" and not loaded.multi_layer
    # a dangling override (the host collapsed the outer skill away) is tolerated everywhere
    dangling = SkillSet.from_documents([SkillDocument(name="x", body="X", overrides="gone")])
    assert dangling.overridden() == {} and [s.name for s in dangling.effective()] == ["x"] and "x" in dangling.render_for_prompt()


def test_content_digest_ignores_purpose_history_and_host_ids():
    a = SkillSet.from_documents([SkillDocument(name="a", body="A", purpose="## Evolution History\n- Iteration 1: x", frontmatter={"id": "row-1", "description": "d"})])
    b = SkillSet.from_documents([SkillDocument(name="a", body="A", purpose="## Evolution History\n- Iteration 7: y", frontmatter={"id": "row-2", "description": "d"})])
    c = SkillSet.from_documents([SkillDocument(name="a", body="A changed", frontmatter={"id": "row-1", "description": "d"})])
    assert a.content_digest() == b.content_digest() != c.content_digest()
    assert a.digest != b.digest  # the full digest still sees every field


def _eval(ref, scores):
    per_task = [TaskOutcome(task_id=f"t{i}", score=s, passed=s >= 0.5) for i, s in enumerate(scores)]
    return Evaluation(ref=ref, score=sum(scores) / len(scores), per_task=per_task)


async def test_paired_gate_rejects_noise_and_accepts_consistent_gains():
    gate = PairedGate(min_tasks=10, resamples=500)
    base = _eval("base", [0.0, 1.0] * 20)
    # one task flips up, one flips down: the mean is unchanged, the bootstrap cannot call it a win
    noisy = _eval("noisy", [1.0, 0.0] + [0.0, 1.0] * 19)
    decision = await gate.decide(base, noisy)
    assert not decision.accepted and decision.feedback["paired_tasks"] == 40 and decision.feedback["win_probability"] < 0.9
    # a small gain on a single task: the paper's rule would accept it, the paired gate does not
    lucky = _eval("lucky", [1.0] + [1.0, 0.0, 1.0] + [0.0, 1.0] * 18)
    assert lucky.score > base.score and not (await gate.decide(base, lucky)).accepted
    # eight tasks fixed, none broken: accepted with a positive CI
    better = _eval("better", [1.0] * 8 + [0.0, 1.0] * 16)
    decision = await gate.decide(base, better)
    assert decision.accepted and decision.feedback["delta_95_ci"][0] > 0 and decision.feedback["mean_delta"] == 0.1
    # too few shared tasks: refuses with a reason rather than guessing
    thin = await gate.decide(_eval("a", [0.0] * 5), _eval("b", [1.0] * 5))
    assert not thin.accepted and thin.feedback["reason"] == "insufficient_paired_tasks"
    # the harness's perfect-score probe never raises and only sets stop from the score
    probe = Evaluation(ref="x", score=1.0)
    assert (await gate.decide(probe, probe)).stop and not (await gate.decide(probe, probe)).accepted
    assert not (await gate.decide(Evaluation(ref="y", score=0.4), Evaluation(ref="y", score=0.4))).stop
    with pytest.raises(ValueError):
        PairedGate(min_win_probability=0.2)
