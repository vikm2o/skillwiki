# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""``skillwiki``: inspect and export a workspace held in a revision store (§3.11).

The paper's artefacts are files you can open; skillwiki's are digest-chained documents. This command line gives an
operator the same visibility without writing code: the current wiki and skills, the skill-impact log, the revision
history and the diff between two accepted skill sets.

    skillwiki --store file:./workspace --workspace default show
    skillwiki --store postgres:postgresql+asyncpg://user:pw@host/db --workspace tenant-a export ./out

Only ``file:`` and ``postgres:`` stores are addressable from the command line; object-store adapters need credentials
a host configures in code.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence

from .skills import SkillSet, unified_diff
from .stores.base import RevisionStore
from .stores.file import FileRevisionStore, export_workspace
from .stores.revision import CHECKPOINTS, SKILLS, TRACES, WIKI, RevisionSkillStore, RevisionWikiStore
from .stores.transfer import seed_workspace
from .wiki import WikiState

KINDS = (WIKI, SKILLS, TRACES, CHECKPOINTS)


class CliError(Exception):
    pass


def open_store(spec: str) -> RevisionStore:
    scheme, _, target = spec.partition(":")
    if scheme == "file" and target:
        return FileRevisionStore(target)
    if scheme == "postgres" and target:
        try:
            from .stores.postgres import PostgresRevisionStore
        except ImportError as exc:  # sqlalchemy missing
            raise CliError("the postgres store needs the extra: pip install 'skillwiki[postgres]'") from exc
        try:
            return PostgresRevisionStore.from_url(target)
        except ModuleNotFoundError as exc:  # driver missing (asyncpg / psycopg)
            raise CliError(f"the database driver for {target.split('://', 1)[0]!r} is not installed: {exc}") from exc
        except Exception as exc:  # a malformed URL: sqlalchemy's ArgumentError and friends
            raise CliError(f"cannot open {target!r}: {exc}. Expected postgres:postgresql+asyncpg://user:pw@host/db") from exc
    raise CliError(f"unsupported store {spec!r}: use file:DIR or postgres:URL")


async def _heads(store: RevisionStore, workspace: str) -> tuple[str | None, WikiState, str | None, SkillSet]:
    wiki_ref, wiki = await RevisionWikiStore(store, workspace).load()
    skills_ref, skills = await RevisionSkillStore(store, workspace).load()
    return wiki_ref, wiki, skills_ref, skills


async def cmd_show(store: RevisionStore, args: argparse.Namespace) -> str:
    wiki_ref, wiki, skills_ref, skills = await _heads(store, args.workspace)
    lines = [
        f"workspace: {args.workspace}",
        f"wiki: {wiki_ref[:12] if wiki_ref else '(empty)'}  iteration {wiki.iteration}  {len(wiki.patterns)} pattern(s)  "
        f"{len(wiki.retired)} retired  {len(wiki.impact)} impact entr{'y' if len(wiki.impact) == 1 else 'ies'}  {wiki.render_size()} chars",
        f"skills: {skills_ref[:12] if skills_ref else '(empty)'}  {len(skills.skills)} skill(s)",
        "",
        "## Skills",
    ]
    hidden = skills.overridden()
    for name in sorted(skills.skills):
        skill = skills.skills[name]
        notes = [f"layer {skill.layer}"] if skills.multi_layer else []
        notes += ["protected"] if skill.protected else []
        notes += [f"overridden by {hidden[name]}"] if name in hidden else []
        notes += [f"overrides {skill.overrides}"] if skill.overrides else []
        lines.append(f"- {name}" + (f" ({'; '.join(notes)})" if notes else "") + f": {skill.description or '(no description)'}")
    lines += ["", "## Wiki index", wiki.render_index(), "", "## Log (tail)", "\n".join(wiki.render_log().splitlines()[-args.tail:] if args.tail > 0 else [])]
    return "\n".join(lines)


async def cmd_export(store: RevisionStore, args: argparse.Namespace) -> str:
    _, wiki, _, skills = await _heads(store, args.workspace)
    written = export_workspace(args.directory, wiki=wiki, skills=skills, full_bodies=args.full_bodies)
    return "\n".join(str(path) for path in written)


async def cmd_impact(store: RevisionStore, args: argparse.Namespace) -> str:
    _, wiki, _, _ = await _heads(store, args.workspace)
    return wiki.render_impact(full_bodies=args.full_bodies)


async def cmd_history(store: RevisionStore, args: argparse.Namespace) -> str:
    kinds = [args.kind] if args.kind else list(KINDS)
    lines = []
    for kind in kinds:
        for revision in await store.list(args.workspace, kind, limit=args.limit):
            meta = json.dumps(revision.meta, sort_keys=True, default=str) if revision.meta else ""
            lines.append(f"{kind:<10} #{revision.seq:<4} {revision.digest[:12]}  {revision.created_at}  {meta}"[:300])
    return "\n".join(lines) if lines else "(no revisions)"


async def cmd_diff(store: RevisionStore, args: argparse.Namespace) -> str:
    revisions = await store.list(args.workspace, SKILLS, limit=1_000_000)  # newest first
    if not revisions:
        raise CliError("no accepted skill sets in this workspace")
    by_digest = {r.digest: r for r in revisions}

    def pick(ref: str | None, default_index: int):
        if ref is None:
            return revisions[default_index] if default_index < len(revisions) else None
        matches = [r for d, r in by_digest.items() if d.startswith(ref)] or [r for r in revisions if str(r.seq) == ref]
        if len(matches) != 1:
            raise CliError(f"skill revision {ref!r} is {'ambiguous' if matches else 'unknown'}")
        return matches[0]

    after = pick(args.to, 0)
    before = pick(getattr(args, "from"), 1)
    before_set = SkillSet.from_document(before.document) if before else SkillSet()
    after_set = SkillSet.from_document(after.document)
    head = f"--- skills #{before.seq} {before.digest[:12]}\n+++ skills #{after.seq} {after.digest[:12]}\n" if before else f"+++ skills #{after.seq} {after.digest[:12]}\n"
    return head + (unified_diff(before_set, after_set) or "(no change)")


async def cmd_transfer(store: RevisionStore, args: argparse.Namespace) -> str:
    digest = await seed_workspace(store, source=args.workspace, target=args.to)
    if digest is None:
        return f"workspace {args.workspace!r} has no wiki to transfer"
    _, wiki = await RevisionWikiStore(store, args.to).load()
    return f"{args.to}: wiki {digest[:12]}  {len(wiki.patterns)} pattern(s)  {len(wiki.retired)} retired"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skillwiki", description="Inspect and export a skillwiki workspace.")
    parser.add_argument("--store", required=True, help="file:DIR or postgres:URL")
    parser.add_argument("--workspace", default="default")
    sub = parser.add_subparsers(dest="command", required=True)
    show = sub.add_parser("show", help="current skills, wiki index and log tail")
    show.add_argument("--tail", type=int, default=10)
    show.set_defaults(run=cmd_show)
    export = sub.add_parser("export", help="write the paper's wiki/ and skills/ layout to a directory")
    export.add_argument("directory")
    export.add_argument("--full-bodies", type=int, default=5)
    export.set_defaults(run=cmd_export)
    impact = sub.add_parser("impact", help="the skill-impact log")
    impact.add_argument("--full-bodies", type=int, default=5)
    impact.set_defaults(run=cmd_impact)
    history = sub.add_parser("history", help="revision chains, newest first")
    history.add_argument("--kind", choices=KINDS)
    history.add_argument("--limit", type=int, default=20)
    history.set_defaults(run=cmd_history)
    diff = sub.add_parser("diff", help="unified diff between two accepted skill sets (default: previous head vs head)")
    diff.add_argument("--from", dest="from", help="digest prefix or seq")
    diff.add_argument("--to", help="digest prefix or seq")
    diff.set_defaults(run=cmd_diff)
    transfer = sub.add_parser("transfer", help="inherit this workspace's wiki into another workspace")
    transfer.add_argument("--to", required=True, help="target workspace")
    transfer.set_defaults(run=cmd_transfer)
    return parser


async def _run(store: RevisionStore, args: argparse.Namespace) -> str:
    try:
        return await args.run(store, args)
    finally:
        engine = getattr(store, "engine", None)  # the postgres store owns an async engine bound to this loop
        if engine is not None:
            await engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        store = open_store(args.store)
        print(asyncio.run(_run(store, args)))
    except CliError as exc:
        print(f"skillwiki: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
