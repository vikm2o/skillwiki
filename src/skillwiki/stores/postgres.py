# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""PostgreSQL revision store (``pip install skillwiki[postgres]``): the recommended production store.

One table, ``skillwiki_revisions``, keyed by ``(workspace, kind, seq)``. The
primary key is what makes the chain fork-free: two writers that both read head
``seq = n`` both try to insert ``n + 1`` and exactly one succeeds. Rows are
never updated or deleted by this module.
"""

from __future__ import annotations

from typing import Any

from ..documents import HeadMoved, Revision

DDL = """
CREATE TABLE IF NOT EXISTS skillwiki_revisions (
    workspace     TEXT        NOT NULL,
    kind          TEXT        NOT NULL,
    seq           INTEGER     NOT NULL,
    digest        CHAR(64)    NOT NULL,
    parent_digest CHAR(64),
    document      JSONB       NOT NULL,
    meta          JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (workspace, kind, seq)
);
CREATE INDEX IF NOT EXISTS ix_skillwiki_revisions_digest ON skillwiki_revisions (workspace, kind, digest);
"""


class PostgresRevisionStore:
    """Works with any SQLAlchemy 2 async engine (``asyncpg`` or ``psycopg`` drivers)."""

    def __init__(self, engine):
        self.engine = engine

    @classmethod
    def from_url(cls, url: str, **engine_kwargs) -> PostgresRevisionStore:
        from sqlalchemy.ext.asyncio import create_async_engine

        return cls(create_async_engine(url, **engine_kwargs))

    async def create_tables(self) -> None:
        from sqlalchemy import text

        async with self.engine.begin() as conn:
            for statement in DDL.strip().split(";"):
                if statement.strip():
                    await conn.execute(text(statement))

    @staticmethod
    def _row(row) -> Revision:
        return Revision(
            workspace=row.workspace,
            kind=row.kind,
            seq=row.seq,
            digest=row.digest,
            parent_digest=row.parent_digest,
            document=row.document,
            created_at=row.created_at.isoformat(),
            meta=row.meta or {},
        )

    async def head(self, workspace: str, kind: str) -> Revision | None:
        from sqlalchemy import text

        async with self.engine.connect() as conn:
            row = (await conn.execute(text(
                "SELECT * FROM skillwiki_revisions WHERE workspace = :w AND kind = :k ORDER BY seq DESC LIMIT 1"
            ), {"w": workspace, "k": kind})).first()
        return self._row(row) if row else None

    async def append(
        self, workspace: str, kind: str, document: dict[str, Any], *, expected_head: str | None, meta: dict[str, Any] | None = None
    ) -> Revision:
        import json

        from sqlalchemy import text
        from sqlalchemy.exc import IntegrityError

        current = await self.head(workspace, kind)
        actual = current.digest if current else None
        if actual != expected_head:
            raise HeadMoved(workspace, kind, expected_head, actual)
        revision = Revision.build(workspace, kind, document, parent=current, meta=meta)
        try:
            async with self.engine.begin() as conn:
                row = (await conn.execute(text(
                    "INSERT INTO skillwiki_revisions (workspace, kind, seq, digest, parent_digest, document, meta) "
                    "VALUES (:w, :k, :s, :d, :p, CAST(:doc AS JSONB), CAST(:meta AS JSONB)) RETURNING created_at"
                ), {"w": workspace, "k": kind, "s": revision.seq, "d": revision.digest, "p": revision.parent_digest,
                    "doc": json.dumps(revision.document, ensure_ascii=False), "meta": json.dumps(revision.meta, ensure_ascii=False)})).first()
        except IntegrityError as exc:
            raise HeadMoved(workspace, kind, expected_head, "another writer appended concurrently") from exc
        return Revision(**{**revision.to_document(), "created_at": row.created_at.isoformat()})

    async def get(self, workspace: str, kind: str, digest: str) -> Revision | None:
        from sqlalchemy import text

        async with self.engine.connect() as conn:
            row = (await conn.execute(text(
                "SELECT * FROM skillwiki_revisions WHERE workspace = :w AND kind = :k AND digest = :d ORDER BY seq LIMIT 1"
            ), {"w": workspace, "k": kind, "d": digest})).first()
        return self._row(row) if row else None

    async def list(self, workspace: str, kind: str, *, limit: int = 50) -> list[Revision]:
        from sqlalchemy import text

        async with self.engine.connect() as conn:
            rows = (await conn.execute(text(
                "SELECT * FROM skillwiki_revisions WHERE workspace = :w AND kind = :k ORDER BY seq DESC LIMIT :n"
            ), {"w": workspace, "k": kind, "n": limit})).all()
        return [self._row(row) for row in rows]


__all__ = ["DDL", "PostgresRevisionStore"]
