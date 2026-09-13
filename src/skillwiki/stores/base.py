# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Store protocols. One primitive (``RevisionStore``) and three layer stores built on it.

A host can implement the three layer protocols directly against its own tables
(a host with its own schema) or implement only ``RevisionStore`` and use the default
``Revision*Store`` classes from :mod:`skillwiki.stores.revision`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..documents import Revision
from ..skills import Proposal, SkillSet
from ..traces import Trace
from ..wiki import WikiState


class RevisionStore(Protocol):
    """An append-only chain per (workspace, kind). ``append`` must refuse to fork: when ``expected_head`` is not
    the digest of the current head (or ``None`` for an empty chain) it raises :class:`skillwiki.documents.HeadMoved`."""

    async def head(self, workspace: str, kind: str) -> Revision | None: ...

    async def append(
        self, workspace: str, kind: str, document: dict[str, Any], *, expected_head: str | None, meta: dict[str, Any] | None = None
    ) -> Revision: ...

    async def get(self, workspace: str, kind: str, digest: str) -> Revision | None: ...

    async def list(self, workspace: str, kind: str, *, limit: int = 50) -> list[Revision]: ...


@dataclass
class Candidate:
    """A materialised candidate skill set S'_k, identified by a host-opaque ``ref`` (a digest, a bundle id, ...)."""

    ref: str
    skill_set: SkillSet
    proposal: Proposal
    meta: dict[str, Any] = field(default_factory=dict)


class WikiStore(Protocol):
    async def load(self) -> tuple[str | None, WikiState]:
        """Current head: (ref, state). A fresh workspace returns (None, WikiState())."""
        ...

    async def save(self, state: WikiState, *, expected_ref: str | None, iteration: int) -> str:
        """Append a new revision and return its ref."""
        ...

    async def invalidated_trace_ids(self) -> set[str]:
        """Trace ids whose evidence the host no longer stands behind (labels corrected, retracted). Default: none."""
        ...


class SkillStore(Protocol):
    """``propose`` MUST be idempotent for a given (iteration, proposal): a resumed run calls it again for the same
    candidate (§3.7), so a host that mints a durable artefact there must key it on the iteration, not append blindly."""

    async def load(self) -> tuple[str | None, SkillSet]: ...

    async def propose(
        self, current_ref: str | None, current: SkillSet, proposal: Proposal, candidate: SkillSet, *, iteration: int, wiki_ref: str | None
    ) -> Candidate:
        """Materialise the candidate so it can be evaluated. Rejected candidates are never promoted."""
        ...

    async def accept(self, candidate: Candidate, *, expected_ref: str | None) -> str:
        """Make the candidate the new head and return its ref."""
        ...


class CheckpointStore(Protocol):
    """Resume support (docs/paper-differences.md §3.7): the harness records where an iteration got to so a restart
    skips the paid stages already done. Hosts without a resume path use :class:`NullCheckpointStore`."""

    async def load(self, iteration: int) -> dict[str, Any] | None:
        """The newest checkpoint recorded for ``iteration``, or ``None``."""
        ...

    async def save(self, iteration: int, payload: dict[str, Any]) -> None: ...


class NullCheckpointStore:
    async def load(self, iteration: int) -> dict[str, Any] | None:
        return None

    async def save(self, iteration: int, payload: dict[str, Any]) -> None:
        return None


class TraceStore(Protocol):
    """The Raw Layer. Immutable; the harness writes each iteration's collected traces once."""

    async def put(self, traces: list[Trace], *, iteration: int) -> None: ...

    async def get(self, trace_id: str) -> Trace | None: ...


class NullTraceStore:
    """For hosts whose traces already live elsewhere (read-only raw layer)."""

    async def put(self, traces: list[Trace], *, iteration: int) -> None:
        return None

    async def get(self, trace_id: str) -> Trace | None:
        return None


__all__ = ["Candidate", "CheckpointStore", "NullCheckpointStore", "NullTraceStore", "RevisionStore", "SkillStore", "TraceStore", "WikiStore"]
