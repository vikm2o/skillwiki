# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
import pytest

from skillwiki import frontmatter
from skillwiki.documents import HeadMoved, Revision, canonical_json, digest
from skillwiki.patch import PatchError, PatchOp, apply_patch, parse_ops
from skillwiki.stores.memory import MemoryRevisionStore


def test_canonical_json_is_order_independent_and_unicode_preserving():
    assert canonical_json({"b": 1, "a": [2, {"d": 1, "c": "é"}]}) == '{"a":[2,{"c":"é","d":1}],"b":1}'
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})


def test_canonical_json_rejects_nan():
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_revision_round_trip_and_verify():
    revision = Revision.build("ws", "wiki", {"k": 1}, parent=None)
    assert Revision.from_document(revision.to_document()) == revision
    revision.verify()
    bad = Revision(**{**revision.to_document(), "digest": "0" * 64})
    with pytest.raises(ValueError):
        bad.verify()


async def test_memory_store_refuses_to_fork():
    store = MemoryRevisionStore()
    first = await store.append("ws", "wiki", {"n": 1}, expected_head=None)
    second = await store.append("ws", "wiki", {"n": 2}, expected_head=first.digest)
    assert second.seq == 1 and second.parent_digest == first.digest
    with pytest.raises(HeadMoved):
        await store.append("ws", "wiki", {"n": 3}, expected_head=first.digest)
    assert [r.seq for r in await store.list("ws", "wiki")] == [1, 0]
    assert (await store.get("ws", "wiki", first.digest)) == first


def test_patch_ops():
    text = "# Title\nline one\nline two\n"
    out = apply_patch(text, parse_ops([{"op": "replace", "target": "line one", "content": "LINE ONE"}]))
    assert "LINE ONE" in out and "line one" not in out
    out = apply_patch(text, [PatchOp("insert_after", "inserted", target="line one")])
    assert out.splitlines() == ["# Title", "line one", "inserted", "line two"]
    out = apply_patch(text, [PatchOp("append", "tail")])
    assert out.endswith("line two\ntail")
    with pytest.raises(PatchError, match="not found"):
        apply_patch(text, [PatchOp("replace", "x", target="missing")])
    with pytest.raises(PatchError, match="ambiguous"):
        apply_patch("a a", [PatchOp("replace", "b", target="a")])
    with pytest.raises(PatchError):
        parse_ops([{"op": "delete"}])
    with pytest.raises(PatchError):
        parse_ops([{"op": "replace", "content": "x"}])


def test_frontmatter_round_trip():
    fields = {"name": "my-skill", "description": "Handles: things, well", "issue_ids": ["a", "b"], "n": 3, "flag": True, "txt": "123"}
    rendered = frontmatter.render(fields)
    parsed, body = frontmatter.split(rendered + "body\n")
    assert parsed == fields and body == "body\n"
    assert frontmatter.split("no frontmatter") == ({}, "no frontmatter")
