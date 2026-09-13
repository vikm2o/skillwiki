# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Seed one workspace's wiki from another's (§3.13): a new project starts from what its organisation already learnt."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import RevisionStore
from .revision import RevisionWikiStore


async def seed_workspace(
    revisions: RevisionStore, *, source: str, target: str, evidence_meta: Callable[[dict[str, Any]], dict[str, Any]] | None = None
) -> str | None:
    """Inherit ``source``'s current wiki into ``target`` and save the result as ``target``'s next wiki revision.

    Returns the new revision digest, or ``None`` when the source has nothing to give. Safe to repeat: patterns the
    target already holds are skipped, and an unchanged target is not re-saved.
    """
    _, source_wiki = await RevisionWikiStore(revisions, source).load()
    if not source_wiki.patterns and not source_wiki.retired:
        return None
    target_store = RevisionWikiStore(revisions, target)
    ref, target_wiki = await target_store.load()
    merged = target_wiki.inherit_from(source_wiki, source=source, evidence_meta=evidence_meta)
    if merged.digest == target_wiki.digest:
        return ref
    return await target_store.save(merged, expected_ref=ref, iteration=merged.iteration)


__all__ = ["seed_workspace"]
