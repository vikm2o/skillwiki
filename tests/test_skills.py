# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
import pytest

from skillwiki.skills import Proposal, ProposalError, SkillDocument, SkillSet, apply_proposal, unified_diff

SKILL_MD = "---\nname: break_loop\ndescription: Stop repeating actions\n---\n# Break loop\n## Instructions\nNever return an item to its origin.\n"


def test_create_patch_retire_and_merge():
    empty = SkillSet()
    created = apply_proposal(
        empty,
        Proposal.from_dict({"action": "create", "name": "break_loop", "skill_md": SKILL_MD, "purpose_md": "## Origin\nfrom pattern loop", "rationale": "r"}),
        iteration=1,
    )
    assert created.skills["break_loop"].description == "Stop repeating actions"
    assert created.to_files()["skills/break_loop/SKILL.md"] == SKILL_MD
    assert empty.skills == {}  # never mutated

    patched = apply_proposal(
        created,
        Proposal.from_dict({"action": "patch", "name": "break_loop", "edits": [{"op": "append", "content": "Each operation ONCE per item."}], "rationale": "new evidence"}),
        iteration=4,
    )
    assert patched.skills["break_loop"].body.endswith("Each operation ONCE per item.")
    assert "Iteration 4: Patched: new evidence" in patched.skills["break_loop"].purpose

    other = apply_proposal(
        patched,
        Proposal.from_dict({"action": "create", "name": "other", "skill_md": SKILL_MD.replace("break_loop", "other"), "purpose_md": "o"}),
        iteration=5,
    )
    merged = apply_proposal(
        other,
        Proposal.from_dict(
            {"action": "create", "name": "loops", "skill_md": SKILL_MD.replace("break_loop", "loops"), "purpose_md": "m", "supersedes": ["break_loop", "other"]}
        ),
        iteration=6,
    )
    assert set(merged.skills) == {"loops"} and "merging: break_loop, other" in merged.skills["loops"].purpose
    retired = apply_proposal(merged, Proposal.from_dict({"action": "retire", "name": "loops"}), iteration=7)
    assert retired.skills == {}
    diff = unified_diff(other, merged)
    assert "--- skills/break_loop/SKILL.md" in diff and "+++ skills/loops/SKILL.md" in diff


def test_protected_skills_and_bad_proposals():
    skills = SkillSet.from_documents([SkillDocument(name="policy", body="x", protected=True)])
    with pytest.raises(ProposalError, match="protected"):
        apply_proposal(skills, Proposal.from_dict({"action": "retire", "name": "policy"}), iteration=1)
    assert apply_proposal(skills, Proposal.from_dict({"action": "retire", "name": "policy"}), iteration=1, allow_protected=True).skills == {}
    with pytest.raises(ProposalError, match="does not exist"):
        apply_proposal(skills, Proposal.from_dict({"action": "patch", "name": "nope", "edits": [{"op": "append", "content": "x"}]}), iteration=1)
    with pytest.raises(ProposalError, match="already exists"):
        apply_proposal(skills, Proposal.from_dict({"action": "create", "name": "policy", "skill_md": SKILL_MD.replace("break_loop", "policy"), "purpose_md": "p"}), iteration=1)
    with pytest.raises(ProposalError, match="does not match"):
        apply_proposal(skills, Proposal.from_dict({"action": "create", "name": "mismatch", "skill_md": SKILL_MD, "purpose_md": "p"}), iteration=1)
    with pytest.raises(ProposalError, match="unknown action"):
        Proposal.from_dict({"action": "delete", "name": "x"})
    with pytest.raises(ProposalError, match="skill_md"):
        Proposal.from_dict({"action": "create", "name": "x"})
    with pytest.raises(ProposalError, match="only valid with action 'create'"):
        Proposal.from_dict({"action": "patch", "name": "policy", "edits": [{"op": "append", "content": "x"}], "supersedes": ["a"]})
    assert Proposal.from_dict({"action": "no_action", "rationale": "fine"}).action == "no_action"


def test_skill_set_document_round_trip():
    skills = SkillSet.from_documents([SkillDocument(name="a", body="b", purpose="p", frontmatter={"description": "d", "issue_ids": ["x"]})])
    assert SkillSet.from_document(skills.to_document()).digest == skills.digest
    with pytest.raises(ProposalError, match="duplicate"):
        SkillSet.from_documents([SkillDocument(name="a", body="1"), SkillDocument(name="a", body="2")])


def test_model_authored_frontmatter_errors_are_proposal_errors_not_crashes():
    """A YAML list or a key render() cannot write back used to escape as a bare ValueError and kill the run."""
    import pytest

    from skillwiki.skills import ProposalError, unified_diff

    listed = "---\nname: foo\ndescription: d\ntags:\n  - a\n  - b\n---\n# Foo\nbody\n"
    with pytest.raises(ProposalError, match="one 'key: value' line"):
        SkillDocument.parse_skill_md(listed, fallback_name="foo")
    spaced = "---\nname: bar\ndescription: d\nWhen to use: always\n---\n# Bar\nbody\n"
    with pytest.raises(ProposalError, match="key 'When to use' is not allowed"):
        SkillDocument.parse_skill_md(spaced, fallback_name="bar")
    # Through apply_proposal the same errors reach the proposer's except-ProposalError, and a valid key still diffs.
    proposal = Proposal.from_dict({"action": "create", "name": "bar", "skill_md": spaced, "purpose_md": "## Origin\nx"})
    with pytest.raises(ProposalError):
        apply_proposal(SkillSet(), proposal, iteration=1)
    ok = Proposal.from_dict({"action": "create", "name": "bar", "skill_md": spaced.replace("When to use", "when_to_use"), "purpose_md": "p"})
    assert "when_to_use: always" in unified_diff(SkillSet(), apply_proposal(SkillSet(), ok, iteration=1))
    # CRLF files parse like LF ones.
    crlf = SkillDocument.parse_skill_md("---\r\nname: baz\r\ndescription: d\r\n---\r\n# Baz\r\nbody\r\n", fallback_name="baz")
    assert crlf.frontmatter == {"description": "d"} and crlf.body == "# Baz\nbody\n"


def test_content_digest_covers_overrides_and_effective_render_hides_overridden_skills():
    from skillwiki.skills import Layer

    base = SkillSet.from_documents([SkillDocument("outer", "rule", layer="global")], layers=[Layer("global", False), Layer("project", True)])
    md = "---\nname: inner\n---\nnew rule\n"
    plain = apply_proposal(base, Proposal.from_dict({"action": "create", "name": "inner", "skill_md": md, "purpose_md": "p"}), iteration=1)
    override = apply_proposal(
        base, Proposal.from_dict({"action": "override", "name": "inner", "overrides": "outer", "skill_md": md, "purpose_md": "p"}), iteration=1
    )
    assert plain.content_digest() != override.content_digest(), "the agent reads different things: 'outer' is hidden in one"
    roles_view, agent_view = override.render_for_prompt(), override.render_for_prompt(effective=True)
    assert "(overridden by inner)" in roles_view and "Skill: outer" in roles_view
    assert "outer" not in agent_view and "Skill: inner" in agent_view and "## Layer" not in agent_view
    assert SkillSet().render_for_prompt(effective=True) == "(no active skills)\n"
