# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
import pytest

from skillwiki.wiki import EvidenceRef, ImpactEntry, IndexEntry, MaintainerUpdate, Pattern, WikiError, WikiState


def _update(**kwargs):
    base = {"create_patterns": [], "update_patterns": [], "update_index": "", "append_log": "did things"}
    return MaintainerUpdate.from_dict({**base, **kwargs})


def test_create_then_patch_pattern_and_index_rules():
    wiki = WikiState()
    wiki = wiki.apply_maintainer_update(
        _update(
            create_patterns=[{"name": "take-examine-move-loop.md", "content": "# Loop\nAgent loops.", "evidence": ["t1", {"trace_id": "t2", "note": "persists"}]}],
            update_index="- [take-examine-move-loop](wiki/patterns/take-examine-move-loop.md): loops; root cause; fix\n- [ghost](wiki/patterns/ghost.md): nope",
        ),
        iteration=1,
        allowed_evidence={"t1", "t2"},
    )
    assert set(wiki.patterns) == {"take-examine-move-loop"}
    assert [e.trace_id for e in wiki.patterns["take-examine-move-loop"].evidence] == ["t1", "t2"]
    # host provenance flows onto citations when the caller passes a mapping
    stamped = WikiState().apply_maintainer_update(
        _update(create_patterns=[{"name": "p", "content": "x", "evidence": ["t1"]}]), iteration=1,
        allowed_evidence={"t1": {"label_revision_ids": ["r1"]}, "t2": {}})
    assert stamped.patterns["p"].evidence[0].meta == {"label_revision_ids": ["r1"]}
    assert WikiState.from_document(stamped.to_document()).patterns["p"].evidence[0].meta == {"label_revision_ids": ["r1"]}
    assert wiki.index == [IndexEntry("take-examine-move-loop", "loops; root cause; fix")]
    assert wiki.log[-1].iteration == 1 and wiki.iteration == 1

    warnings: list[str] = []
    wiki2 = wiki.apply_maintainer_update(
        _update(update_patterns=[{"name": "take-examine-move-loop", "edits": [{"op": "append", "content": "Fix: stop."}], "evidence": ["t3"]}]),
        iteration=2,
        allowed_evidence={"t3"},
        warnings=warnings,
    )
    page = wiki2.patterns["take-examine-move-loop"]
    assert page.body.endswith("Fix: stop.") and page.updated_iteration == 2 and page.created_iteration == 1
    assert [e.trace_id for e in page.evidence] == ["t1", "t2", "t3"]
    # empty update_index: the harness restores every pattern with its title rather than losing the catalogue
    assert wiki2.index[0].pattern_id == "take-examine-move-loop" and any("omitted" in w for w in warnings)
    # the original state is untouched (never rolled back, never mutated in place)
    assert wiki.patterns["take-examine-move-loop"].body == "# Loop\nAgent loops."


def test_citations_must_come_from_the_sample_and_duplicates_are_rejected():
    wiki = WikiState()
    with pytest.raises(WikiError, match="not in this iteration's sample"):
        wiki.apply_maintainer_update(_update(create_patterns=[{"name": "p", "content": "x", "evidence": ["nope"]}]), iteration=1, allowed_evidence={"t1"})
    wiki = wiki.apply_maintainer_update(_update(create_patterns=[{"name": "p", "content": "x", "evidence": ["t1"]}]), iteration=1, allowed_evidence={"t1"})
    with pytest.raises(WikiError, match="already exists"):
        wiki.apply_maintainer_update(_update(create_patterns=[{"name": "p", "content": "y", "evidence": ["t1"]}]), iteration=2, allowed_evidence={"t1"})
    with pytest.raises(WikiError, match="target not found"):
        wiki.apply_maintainer_update(
            _update(update_patterns=[{"name": "p", "edits": [{"op": "replace", "target": "zzz", "content": "y"}]}]), iteration=2, allowed_evidence=set()
        )
    with pytest.raises(WikiError, match="required strings"):
        MaintainerUpdate.from_dict({"create_patterns": []})


def test_mark_stale_moves_citations_without_deleting():
    pattern = Pattern(id="p", title="P", body="b", evidence=[EvidenceRef("t1"), EvidenceRef("t2")])
    wiki = WikiState(patterns={"p": pattern})
    stale = wiki.mark_stale({"t2"})
    assert [e.trace_id for e in stale.patterns["p"].evidence] == ["t1"]
    assert [e.trace_id for e in stale.patterns["p"].stale_evidence] == ["t2"]
    assert wiki.mark_stale(set()) is wiki


def test_document_and_markdown_round_trips_match_the_paper_layout():
    wiki = WikiState()
    wiki = wiki.apply_maintainer_update(
        _update(
            create_patterns=[{"name": "loop", "title": "Loop", "content": "# Loop\nbody", "evidence": ["t1"]}],
            update_index="- [loop](wiki/patterns/loop.md): P + RC + fix",
        ),
        iteration=1,
        allowed_evidence={"t1"},
    )
    wiki.record_impact(ImpactEntry(iteration=1, outcome="rejected", action="create", target="s", diff="--- a\n+++ b", validation={"score": 0.7}, proposal_body="BODY"))
    assert WikiState.from_document(wiki.to_document()).to_document() == wiki.to_document()
    files = wiki.to_markdown_files()
    assert set(files) == {"wiki/index.md", "wiki/log.md", "wiki/skill-impact.md", "wiki/patterns/loop.md"}
    assert "- [loop](wiki/patterns/loop.md): P + RC + fix" in files["wiki/index.md"]
    assert "REJECTED" in files["wiki/skill-impact.md"] and "BODY" in files["wiki/skill-impact.md"]
    rebuilt = WikiState.from_markdown_files(files, state=wiki)
    assert rebuilt.patterns["loop"].to_dict() == wiki.patterns["loop"].to_dict()
    assert rebuilt.index == wiki.index


def test_impact_keeps_full_bodies_only_for_recent_rejections():
    wiki = WikiState()
    for i in range(1, 9):
        wiki.record_impact(ImpactEntry(iteration=i, outcome="rejected", action="create", target=f"s{i}", diff="d", validation={}, proposal_body=f"BODY{i}"))
    text = wiki.render_impact(full_bodies=2)
    assert "BODY8" in text and "BODY7" in text and "BODY6" not in text
