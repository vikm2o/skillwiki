# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Filesystem revision store and the paper's on-disk workspace rendering.

For local runs, examples and tests. Not for clustered deployments: there is no
cross-host locking, only atomic renames on one filesystem.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from ..documents import HeadMoved, Revision
from ..skills import SkillSet
from ..wiki import WikiState

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _segment(value: str) -> str:
    return _SAFE.sub("_", value) or "_"


class FileRevisionStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._lock = asyncio.Lock()

    def _dir(self, workspace: str, kind: str) -> Path:
        return self.root / _segment(workspace) / _segment(kind)

    def _read_head(self, directory: Path) -> Revision | None:
        head = directory / "HEAD"
        if not head.exists():
            return None
        path = directory / head.read_text().strip()
        return Revision.from_document(json.loads(path.read_text()))

    async def head(self, workspace: str, kind: str) -> Revision | None:
        return await asyncio.to_thread(self._read_head, self._dir(workspace, kind))

    async def append(
        self, workspace: str, kind: str, document: dict[str, Any], *, expected_head: str | None, meta: dict[str, Any] | None = None
    ) -> Revision:
        async with self._lock:
            return await asyncio.to_thread(self._append, workspace, kind, document, expected_head, meta)

    def _append(self, workspace, kind, document, expected_head, meta) -> Revision:
        directory = self._dir(workspace, kind)
        directory.mkdir(parents=True, exist_ok=True)
        current = self._read_head(directory)
        actual = current.digest if current else None
        if actual != expected_head:
            raise HeadMoved(workspace, kind, expected_head, actual)
        revision = Revision.build(workspace, kind, document, parent=current, meta=meta)
        name = f"{revision.seq:06d}-{revision.digest}.json"
        (directory / name).write_text(json.dumps(revision.to_document(), ensure_ascii=False, indent=1))
        tmp = directory / "HEAD.tmp"
        tmp.write_text(name)
        os.replace(tmp, directory / "HEAD")  # atomic on POSIX and Windows
        return revision

    async def get(self, workspace: str, kind: str, digest: str) -> Revision | None:
        for revision in await self.list(workspace, kind, limit=1_000_000):
            if revision.digest == digest:
                return revision
        return None

    async def list(self, workspace: str, kind: str, *, limit: int = 50) -> list[Revision]:
        directory = self._dir(workspace, kind)
        if not directory.exists():
            return []
        names = sorted((p for p in directory.glob("*.json")), key=lambda p: p.name, reverse=True)
        return [Revision.from_document(json.loads(p.read_text())) for p in names[:limit]]


def export_workspace(root: str | Path, *, wiki: WikiState, skills: SkillSet, full_bodies: int = 5) -> list[Path]:
    """Write the paper's ``wiki/`` and ``skills/`` directories for humans and diff tools. Returns the written paths."""
    base = Path(root)
    written: list[Path] = []
    for relative, text in {**wiki.to_markdown_files(full_bodies=full_bodies), **skills.to_files()}.items():
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        written.append(path)
    return written


__all__ = ["FileRevisionStore", "export_workspace"]
