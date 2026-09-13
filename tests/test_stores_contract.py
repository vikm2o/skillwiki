# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Every RevisionStore adapter passes the same contract: ordered chain, digest identity, refusal to fork."""

import os

import pytest

from skillwiki.documents import HeadMoved
from skillwiki.skills import SkillDocument, SkillSet
from skillwiki.stores.blob import BlobRevisionStore, MemoryBlobStore
from skillwiki.stores.file import FileRevisionStore, export_workspace
from skillwiki.stores.memory import MemoryRevisionStore
from skillwiki.wiki import Pattern, WikiState


async def contract(store):
    assert await store.head("ws", "wiki") is None
    first = await store.append("ws", "wiki", {"n": 1, "text": "é"}, expected_head=None, meta={"iteration": 1})
    assert first.seq == 0 and first.parent_digest is None and first.meta == {"iteration": 1}
    second = await store.append("ws", "wiki", {"n": 2}, expected_head=first.digest)
    assert second.seq == 1 and second.parent_digest == first.digest
    head = await store.head("ws", "wiki")
    assert head.digest == second.digest and head.document == {"n": 2}
    head.verify()
    with pytest.raises(HeadMoved):
        await store.append("ws", "wiki", {"n": 3}, expected_head=first.digest)
    with pytest.raises(HeadMoved):
        await store.append("ws", "wiki", {"n": 3}, expected_head=None)
    assert [r.seq for r in await store.list("ws", "wiki")] == [1, 0]
    assert [r.seq for r in await store.list("ws", "wiki", limit=1)] == [1]
    assert (await store.get("ws", "wiki", first.digest)).document == {"n": 1, "text": "é"}
    assert await store.get("ws", "wiki", "0" * 64) is None
    # chains are independent per (workspace, kind)
    assert await store.head("ws", "skills") is None and await store.head("other", "wiki") is None
    other = await store.append("other", "wiki", {"n": 1}, expected_head=None)
    assert other.seq == 0


async def test_memory_store_contract():
    await contract(MemoryRevisionStore())


async def test_file_store_contract_and_export(tmp_path):
    store = FileRevisionStore(tmp_path / "revisions")
    await contract(store)
    assert (tmp_path / "revisions" / "ws" / "wiki" / "HEAD").read_text().startswith("000001-")
    wiki = WikiState(patterns={"p": Pattern(id="p", title="P", body="# P\nbody")})
    skills = SkillSet.from_documents([SkillDocument(name="s", body="do it", purpose="why")])
    written = export_workspace(tmp_path / "workspace", wiki=wiki, skills=skills)
    names = sorted(str(p.relative_to(tmp_path / "workspace")) for p in written)
    assert names == ["skills/s/PURPOSE.md", "skills/s/SKILL.md", "wiki/index.md", "wiki/log.md", "wiki/patterns/p.md", "wiki/skill-impact.md"]


async def test_blob_store_contract_and_race():
    blobs = MemoryBlobStore()
    store = BlobRevisionStore(blobs, prefix="p")
    await contract(store)
    # A concurrent writer that advanced HEAD between our read and our write loses cleanly.
    head = await store.head("ws", "wiki")
    racer = BlobRevisionStore(blobs, prefix="p")
    await racer.append("ws", "wiki", {"n": 99}, expected_head=head.digest)
    with pytest.raises(HeadMoved):
        await store.append("ws", "wiki", {"n": 100}, expected_head=head.digest)
    assert (await store.head("ws", "wiki")).document == {"n": 99}


@pytest.mark.postgres
async def test_postgres_store_contract():
    pytest.importorskip("asyncpg")
    from testcontainers.postgres import PostgresContainer

    from skillwiki.stores.postgres import PostgresRevisionStore

    if os.environ.get("SKILLWIKI_SKIP_DOCKER"):
        pytest.skip("SKILLWIKI_SKIP_DOCKER set")
    with PostgresContainer("postgres:16-alpine", driver="asyncpg") as pg:
        store = PostgresRevisionStore.from_url(pg.get_connection_url())
        await store.create_tables()
        await contract(store)
        # The primary key refuses the fork even when the head check is bypassed by a racing writer.
        head = await store.head("ws", "wiki")
        racer = PostgresRevisionStore(store.engine)
        await racer.append("ws", "wiki", {"n": 99}, expected_head=head.digest)
        with pytest.raises(HeadMoved):
            await store.append("ws", "wiki", {"n": 100}, expected_head=head.digest)
        await store.engine.dispose()


@pytest.mark.s3
async def test_s3_store_contract_with_moto():
    boto3 = pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")
    from skillwiki.stores.s3 import S3BlobStore

    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="skillwiki-test")
        blobs = S3BlobStore("skillwiki-test", client=client)
        store = BlobRevisionStore(blobs)
        await contract(store)
        # conditional-put semantics on the emulator
        assert await blobs.put_if_absent("x/HEAD", b"1") is True
        assert await blobs.put_if_absent("x/HEAD", b"2") is False
        _, tag = await blobs.get("x/HEAD")
        assert await blobs.put_if_match("x/HEAD", b"3", tag=tag) is True
        assert await blobs.put_if_match("x/HEAD", b"4", tag=tag) is False
