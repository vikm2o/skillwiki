# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Default layer stores over any :class:`RevisionStore`."""

from __future__ import annotations

from typing import Any

from ..documents import SCHEMA_VERSION, Revision
from ..skills import Proposal, SkillSet
from ..traces import Trace
from ..wiki import WikiState
from .base import Candidate, RevisionStore

WIKI, SKILLS, TRACES, CHECKPOINTS = "wiki", "skills", "raw", "checkpoint"


class RevisionWikiStore:
    def __init__(self, revisions: RevisionStore, workspace: str):
        self.revisions, self.workspace = revisions, workspace

    async def load(self) -> tuple[str | None, WikiState]:
        head = await self.revisions.head(self.workspace, WIKI)
        if head is None:
            return None, WikiState()
        head.verify()
        return head.digest, WikiState.from_document(head.document)

    async def save(self, state: WikiState, *, expected_ref: str | None, iteration: int) -> str:
        revision = await self.revisions.append(
            self.workspace, WIKI, state.to_document(), expected_head=expected_ref, meta={"iteration": iteration}
        )
        return revision.digest

    async def invalidated_trace_ids(self) -> set[str]:
        return set()

    async def history(self, *, limit: int = 50) -> list[Revision]:
        return await self.revisions.list(self.workspace, WIKI, limit=limit)


class RevisionSkillStore:
    """Candidates are held in memory until accepted; only accepted skill sets join the chain (paper §3.2.4)."""

    def __init__(self, revisions: RevisionStore, workspace: str):
        self.revisions, self.workspace = revisions, workspace

    async def load(self) -> tuple[str | None, SkillSet]:
        head = await self.revisions.head(self.workspace, SKILLS)
        if head is None:
            return None, SkillSet()
        head.verify()
        return head.digest, SkillSet.from_document(head.document)

    async def propose(
        self, current_ref: str | None, current: SkillSet, proposal: Proposal, candidate: SkillSet, *, iteration: int, wiki_ref: str | None
    ) -> Candidate:
        return Candidate(ref=candidate.digest, skill_set=candidate, proposal=proposal, meta={"iteration": iteration, "wiki_ref": wiki_ref})

    async def accept(self, candidate: Candidate, *, expected_ref: str | None) -> str:
        revision = await self.revisions.append(
            self.workspace,
            SKILLS,
            candidate.skill_set.to_document(),
            expected_head=expected_ref,
            meta={**candidate.meta, "proposal": candidate.proposal.to_dict()},
        )
        return revision.digest

    async def seed(self, skill_set: SkillSet) -> str:
        """Write an onboarded baseline as revision 0 of an empty chain."""
        revision = await self.revisions.append(self.workspace, SKILLS, skill_set.to_document(), expected_head=None, meta={"origin": "onboarded"})
        return revision.digest


class RevisionTraceStore:
    """Each iteration's traces become one immutable ``raw`` revision; individual traces are looked up by id."""

    def __init__(self, revisions: RevisionStore, workspace: str):
        self.revisions, self.workspace = revisions, workspace

    async def put(self, traces: list[Trace], *, iteration: int) -> None:
        head = await self.revisions.head(self.workspace, TRACES)
        document: dict[str, Any] = {"schema": SCHEMA_VERSION, "iteration": iteration, "traces": [t.to_document() for t in traces]}
        await self.revisions.append(self.workspace, TRACES, document, expected_head=head.digest if head else None, meta={"iteration": iteration})

    async def get(self, trace_id: str) -> Trace | None:
        for revision in await self.revisions.list(self.workspace, TRACES, limit=10_000):
            for value in revision.document.get("traces", []):
                if value.get("id") == trace_id:
                    return Trace.from_document(value)
        return None


class RevisionCheckpointStore:
    """Each save appends one ``checkpoint`` revision; ``load`` returns the newest one for the iteration."""

    def __init__(self, revisions: RevisionStore, workspace: str):
        self.revisions, self.workspace = revisions, workspace

    async def load(self, iteration: int) -> dict[str, Any] | None:
        revisions = sorted(await self.revisions.list(self.workspace, CHECKPOINTS, limit=200), key=lambda r: -r.seq)
        for revision in revisions:
            if revision.document.get("iteration") == iteration:
                return dict(revision.document.get("payload", {}))
        return None

    async def save(self, iteration: int, payload: dict[str, Any]) -> None:
        head = await self.revisions.head(self.workspace, CHECKPOINTS)
        document: dict[str, Any] = {"schema": SCHEMA_VERSION, "iteration": iteration, "payload": payload}
        await self.revisions.append(self.workspace, CHECKPOINTS, document, expected_head=head.digest if head else None, meta={"iteration": iteration})


__all__ = ["RevisionCheckpointStore", "RevisionSkillStore", "RevisionTraceStore", "RevisionWikiStore"]
