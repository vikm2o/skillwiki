# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""The Wiki Layer: patterns, index, evolution log and skill-impact tracker.

Storage is a JSON document (:meth:`WikiState.to_document`). The paper's file
layout (``wiki/index.md``, ``wiki/log.md``, ``wiki/skill-impact.md``,
``wiki/patterns/<id>.md``) is produced by :meth:`WikiState.to_markdown_files`
and read back by :meth:`WikiState.from_markdown_files`; the two are inverses.

The wiki is never rolled back. Evidence that the host later invalidates is
recorded on the pattern as ``stale_evidence`` so the maintainer can revise or
retire the claim, instead of the claim silently continuing to rest on it.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import frontmatter
from .documents import SCHEMA_VERSION, digest, now_iso
from .patch import PatchError, PatchOp, apply_patch, parse_ops

PATTERN_ID = re.compile(r"^[a-z0-9][a-z0-9\-_]{0,79}$")


class WikiError(ValueError):
    """A maintainer update was rejected; the message is safe to show a model."""


def normalise_pattern_id(name: str) -> str:
    name = name.strip()
    if name.endswith(".md"):
        name = name[:-3]
    name = name.removeprefix("wiki/patterns/").removeprefix("patterns/")
    return name


@dataclass
class EvidenceRef:
    """A citation of a raw trace. ``meta`` is opaque host provenance (labels, revisions, assets)."""

    trace_id: str
    note: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"trace_id": self.trace_id, "note": self.note, "meta": self.meta}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvidenceRef:
        return cls(trace_id=value["trace_id"], note=value.get("note", ""), meta=dict(value.get("meta", {})))


@dataclass
class Pattern:
    id: str
    title: str
    body: str
    evidence: list[EvidenceRef] = field(default_factory=list)
    stale_evidence: list[EvidenceRef] = field(default_factory=list)
    created_iteration: int = 0
    updated_iteration: int = 0
    inherited_from: str | None = None  # the workspace this pattern was transferred from (§3.13), if any

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "body": self.body,
            "evidence": [e.to_dict() for e in self.evidence],
            "stale_evidence": [e.to_dict() for e in self.stale_evidence],
            **({"inherited_from": self.inherited_from} if self.inherited_from else {}),
            "created_iteration": self.created_iteration,
            "updated_iteration": self.updated_iteration,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Pattern:
        return cls(
            id=value["id"],
            title=value.get("title", value["id"]),
            body=value.get("body", ""),
            evidence=[EvidenceRef.from_dict(e) for e in value.get("evidence", [])],
            stale_evidence=[EvidenceRef.from_dict(e) for e in value.get("stale_evidence", [])],
            created_iteration=int(value.get("created_iteration", 0)),
            updated_iteration=int(value.get("updated_iteration", 0)),
            inherited_from=value.get("inherited_from") or None,
        )

    def to_markdown(self) -> str:
        fields = {
            "id": self.id,
            "title": self.title,
            **({"inherited_from": self.inherited_from} if self.inherited_from else {}),
            "created_iteration": self.created_iteration,
            "updated_iteration": self.updated_iteration,
            "evidence": [e.to_dict() for e in self.evidence],
            "stale_evidence": [e.to_dict() for e in self.stale_evidence],
        }
        return frontmatter.render(fields) + self.body

    @classmethod
    def from_markdown(cls, text: str, *, fallback_id: str) -> Pattern:
        fields, body = frontmatter.split(text)
        return cls.from_dict({**fields, "id": fields.get("id", fallback_id), "body": body})


@dataclass
class IndexEntry:
    pattern_id: str
    description: str

    def render(self) -> str:
        return f"- [{self.pattern_id}](wiki/patterns/{self.pattern_id}.md): {self.description}"

    _LINE = re.compile(r"^- \[(?P<id>[^\]]+)\]\(wiki/patterns/(?P=id)\.md\): (?P<desc>.*)$")

    @classmethod
    def parse(cls, line: str) -> IndexEntry | None:
        match = cls._LINE.match(line.strip())
        return cls(pattern_id=match["id"], description=match["desc"]) if match else None


@dataclass
class LogEntry:
    iteration: int
    text: str
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {"iteration": self.iteration, "text": self.text, "created_at": self.created_at}


@dataclass
class ImpactEntry:
    """One gated proposal (paper §3.2.4): what was proposed, its diff, the validation result and the outcome."""

    iteration: int
    outcome: str  # accepted | rejected | no_action | onboarded
    action: str  # create | patch | retire | no_action | baseline
    target: str | None
    diff: str
    validation: dict[str, Any]
    rationale: str = ""
    proposal_body: str | None = None  # full content of the proposed skill, retained for the most recent rejections
    citations: list[str] = field(default_factory=list)  # pattern ids the proposal relied on (keeps them prunable-safe)
    candidate_digest: str | None = None  # SkillSet.content_digest() of the candidate; lets the harness refuse repeats
    layer: str | None = None  # layer the change landed in
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "outcome": self.outcome,
            "action": self.action,
            "target": self.target,
            "diff": self.diff,
            "validation": self.validation,
            "rationale": self.rationale,
            "proposal_body": self.proposal_body,
            "citations": list(self.citations),
            "candidate_digest": self.candidate_digest,
            "layer": self.layer,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ImpactEntry:
        entry = cls(**{k: value.get(k) for k in cls.__dataclass_fields__ if k in value})
        entry.citations = list(entry.citations or [])
        return entry

    def cited_patterns(self) -> set[str]:
        return set(self.citations)


@dataclass
class MaintainerUpdate:
    """The Wiki Maintainer's incremental-edit output (paper App. E.2)."""

    create_patterns: list[dict[str, Any]] = field(default_factory=list)
    update_patterns: list[dict[str, Any]] = field(default_factory=list)
    update_index: str = ""
    append_log: str = ""

    @classmethod
    def from_dict(cls, value: Any) -> MaintainerUpdate:
        if not isinstance(value, dict):
            raise WikiError("maintainer output must be a JSON object")
        unknown = set(value) - {"create_patterns", "update_patterns", "update_index", "append_log"}
        if unknown:
            raise WikiError(f"unexpected keys in maintainer output: {sorted(unknown)}")
        if not isinstance(value.get("update_index"), str) or not isinstance(value.get("append_log"), str):
            raise WikiError("'update_index' and 'append_log' are required strings")
        for key in ("create_patterns", "update_patterns"):
            if not isinstance(value.get(key, []), list):
                raise WikiError(f"'{key}' must be a list")
        return cls(
            create_patterns=list(value.get("create_patterns", [])),
            update_patterns=list(value.get("update_patterns", [])),
            update_index=value["update_index"],
            append_log=value["append_log"],
        )


@dataclass
class RetiredPattern:
    """A pattern the pruner retired or merged away. Nothing is deleted: title, body and evidence stay readable."""

    id: str
    title: str
    body: str
    evidence: list[EvidenceRef]
    stale_evidence: list[EvidenceRef]
    reason: str
    retired_iteration: int
    merged_into: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "body": self.body,
            "evidence": [e.to_dict() for e in self.evidence],
            "stale_evidence": [e.to_dict() for e in self.stale_evidence],
            "reason": self.reason,
            "retired_iteration": self.retired_iteration,
            "merged_into": self.merged_into,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RetiredPattern:
        return cls(
            id=value["id"],
            title=value.get("title", value["id"]),
            body=value.get("body", ""),
            evidence=[EvidenceRef.from_dict(e) for e in value.get("evidence", [])],
            stale_evidence=[EvidenceRef.from_dict(e) for e in value.get("stale_evidence", [])],
            reason=str(value.get("reason", "")),
            retired_iteration=int(value.get("retired_iteration", 0)),
            merged_into=value.get("merged_into"),
        )


@dataclass
class PruneUpdate:
    """The Wiki Pruner's output (docs/paper-differences.md §3.6): merges and retirements, both reversible in the log."""

    merges: list[dict[str, Any]] = field(default_factory=list)  # {"into": id, "from": [ids], "title"?: str, "content"?: str}
    retire: list[dict[str, Any]] = field(default_factory=list)  # {"name": id, "reason": str}
    append_log: str = ""

    @classmethod
    def from_dict(cls, value: Any) -> PruneUpdate:
        if not isinstance(value, dict):
            raise WikiError("pruner output must be a JSON object")
        unknown = set(value) - {"merges", "retire", "append_log"}
        if unknown:
            raise WikiError(f"unexpected keys in pruner output: {sorted(unknown)}")
        for key in ("merges", "retire"):
            if not isinstance(value.get(key, []), list):
                raise WikiError(f"'{key}' must be a list")
        if not isinstance(value.get("append_log", ""), str):
            raise WikiError("'append_log' must be a string")
        return cls(merges=list(value.get("merges", [])), retire=list(value.get("retire", [])), append_log=value.get("append_log", ""))

    @property
    def empty(self) -> bool:
        return not self.merges and not self.retire


@dataclass
class WikiState:
    patterns: dict[str, Pattern] = field(default_factory=dict)
    index: list[IndexEntry] = field(default_factory=list)
    log: list[LogEntry] = field(default_factory=list)
    impact: list[ImpactEntry] = field(default_factory=list)
    iteration: int = 0
    seen_trace_ids: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    retired: dict[str, RetiredPattern] = field(default_factory=dict)

    # -- persistence -------------------------------------------------------------------------------------------------

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "iteration": self.iteration,
            "patterns": {pid: p.to_dict() for pid, p in sorted(self.patterns.items())},
            "index": [{"pattern_id": e.pattern_id, "description": e.description} for e in self.index],
            "log": [e.to_dict() for e in self.log],
            "impact": [e.to_dict() for e in self.impact],
            "seen_trace_ids": list(self.seen_trace_ids),
            "meta": self.meta,
            "retired": {pid: r.to_dict() for pid, r in sorted(self.retired.items())},
        }

    @classmethod
    def from_document(cls, value: dict[str, Any]) -> WikiState:
        return cls(
            patterns={pid: Pattern.from_dict(p) for pid, p in value.get("patterns", {}).items()},
            index=[IndexEntry(e["pattern_id"], e["description"]) for e in value.get("index", [])],
            log=[LogEntry(**e) for e in value.get("log", [])],
            impact=[ImpactEntry.from_dict(e) for e in value.get("impact", [])],
            iteration=int(value.get("iteration", 0)),
            seen_trace_ids=list(value.get("seen_trace_ids", [])),
            meta=dict(value.get("meta", {})),
            retired={pid: RetiredPattern.from_dict(r) for pid, r in value.get("retired", {}).items()},
        )

    @property
    def digest(self) -> str:
        return digest(self.to_document())

    def copy(self) -> WikiState:
        return copy.deepcopy(self)

    # -- rendering (paper layout) ---------------------------------------------------------------------------------------

    def render_index(self) -> str:
        return "# Pattern index\n\n" + "\n".join(e.render() for e in self.index) + ("\n" if self.index else "")

    def render_log(self) -> str:
        lines = ["# Evolution log", ""]
        for entry in self.log:
            lines.append(f"## Iteration {entry.iteration}")
            lines.append(entry.text.rstrip())
            lines.append("")
        return "\n".join(lines)

    def render_impact(self, *, full_bodies: int = 5) -> str:
        """skill-impact.md: full proposal bodies for the most recent ``full_bodies`` rejections, diffs for the rest."""
        lines = ["# Skill impact", ""]
        rejected_indices = [i for i, e in enumerate(self.impact) if e.outcome == "rejected"]
        keep_body = set(rejected_indices[-full_bodies:]) if full_bodies else set()
        for i, entry in enumerate(self.impact):
            head = f"## Iteration {entry.iteration}: {entry.action} {entry.target or ''}".rstrip()
            if entry.layer:
                head += f" [layer {entry.layer}]"
            lines.append(f"{head} -- {entry.outcome.upper()}")
            if entry.rationale:
                lines.append(f"Rationale: {entry.rationale}")
            if entry.validation:
                summary = ", ".join(f"{k}={v}" for k, v in sorted(entry.validation.items()) if not isinstance(v, (dict, list)))
                lines.append(f"Validation: {summary}" if summary else "Validation: (structured)")
            if entry.diff:
                lines.append("```diff")
                lines.append(entry.diff.rstrip())
                lines.append("```")
            if i in keep_body and entry.proposal_body:
                lines.append("Rejected proposal content:")
                lines.append("````")
                lines.append(entry.proposal_body.rstrip())
                lines.append("````")
            lines.append("")
        return "\n".join(lines)

    def to_markdown_files(self, *, full_bodies: int = 5) -> dict[str, str]:
        files = {
            "wiki/index.md": self.render_index(),
            "wiki/log.md": self.render_log(),
            "wiki/skill-impact.md": self.render_impact(full_bodies=full_bodies),
        }
        for pid, pattern in sorted(self.patterns.items()):
            files[f"wiki/patterns/{pid}.md"] = pattern.to_markdown()
        if self.retired:
            files["wiki/retired.md"] = self.render_retired()
        return files

    def render_retired(self) -> str:
        """retired.md: what pruning removed from the index and why. Bodies are kept so nothing is lost."""
        lines = ["# Retired patterns", ""]
        for pid, item in sorted(self.retired.items(), key=lambda kv: (kv[1].retired_iteration, kv[0])):
            head = f"## {pid} (iteration {item.retired_iteration})"
            if item.merged_into:
                head += f" -- merged into {item.merged_into}"
            lines += [head, f"Reason: {item.reason}", f"Evidence: {len(item.evidence)} live, {len(item.stale_evidence)} stale", "", item.body.rstrip(), ""]
        return "\n".join(lines)

    def render_size(self) -> int:
        """Characters the maintainer would be shown for the whole wiki (index, log tail excluded, every pattern page)."""
        return len(self.render_index()) + sum(len(p.to_markdown()) for p in self.patterns.values())

    def protected_pattern_ids(self) -> set[str]:
        """Patterns cited by an accepted skill change. The pruner may not retire or merge these away."""
        return {pid for entry in self.impact if entry.outcome == "accepted" for pid in entry.citations if pid in self.patterns}

    @classmethod
    def from_markdown_files(cls, files: dict[str, str], *, state: WikiState | None = None) -> WikiState:
        """Rebuild patterns and index from rendered files. Log and impact entries are structured records and
        are taken from ``state`` when given (the markdown renderings of those two files are summaries)."""
        result = state.copy() if state else cls()
        result.patterns = {}
        for path, text in files.items():
            if path.startswith("wiki/patterns/") and path.endswith(".md"):
                pid = path[len("wiki/patterns/") : -3]
                result.patterns[pid] = Pattern.from_markdown(text, fallback_id=pid)
        if "wiki/index.md" in files:
            result.index = [e for e in (IndexEntry.parse(line) for line in files["wiki/index.md"].splitlines()) if e]
        return result

    # -- maintenance ----------------------------------------------------------------------------------------------

    def apply_maintainer_update(
        self,
        update: MaintainerUpdate,
        *,
        iteration: int,
        allowed_evidence: set[str] | dict[str, dict[str, Any]],
        warnings: list[str] | None = None,
    ) -> WikiState:
        """Return a new state with the maintainer's incremental edits applied (paper §3.2.2).

        Citations must name traces from this iteration's sample. When ``allowed_evidence`` is a mapping, each
        trace's opaque host ``meta`` is copied onto the citation, so provenance (label ids, assets) rides along.
        Unknown index entries are dropped with a warning rather than failing the iteration; patch failures are
        errors the caller can feed back to the model.
        """
        state = self.copy()
        warnings = warnings if warnings is not None else []
        if not isinstance(allowed_evidence, dict):
            allowed_evidence = dict.fromkeys(allowed_evidence, {})
        for item in update.create_patterns:
            pid = normalise_pattern_id(str(item.get("name", "")))
            if not PATTERN_ID.match(pid):
                raise WikiError(f"invalid pattern name {item.get('name')!r}: use lowercase letters, digits, '-' or '_'")
            if pid in state.patterns:
                raise WikiError(f"pattern {pid!r} already exists; use update_patterns to add evidence")
            if pid in state.retired:
                raise WikiError(f"pattern {pid!r} was retired ({state.retired[pid].reason}); do not recreate it, extend the surviving pattern")
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                raise WikiError(f"pattern {pid!r} needs non-empty string 'content'")
            evidence = _parse_evidence(item.get("evidence", []), allowed_evidence, pid)
            title = str(item.get("title") or _title_from_body(content) or pid)
            state.patterns[pid] = Pattern(
                id=pid, title=title, body=content, evidence=evidence, created_iteration=iteration, updated_iteration=iteration
            )
        for item in update.update_patterns:
            pid = normalise_pattern_id(str(item.get("name", "")))
            if pid not in state.patterns:
                raise WikiError(f"cannot update unknown pattern {pid!r}")
            pattern = state.patterns[pid]
            if item.get("edits"):
                try:
                    pattern.body = apply_patch(pattern.body, parse_ops(item["edits"]))
                except PatchError as exc:
                    raise WikiError(f"pattern {pid!r}: {exc}") from exc
            for ref in _parse_evidence(item.get("evidence", []), allowed_evidence, pid):
                if all(ref.trace_id != e.trace_id for e in pattern.evidence):
                    pattern.evidence.append(ref)
            if "title" in item and isinstance(item["title"], str) and item["title"].strip():
                pattern.title = item["title"].strip()
            pattern.updated_iteration = iteration
        entries: list[IndexEntry] = []
        for line in update.update_index.splitlines():
            entry = IndexEntry.parse(line)
            if entry is None:
                if line.strip().startswith("- ["):
                    warnings.append(f"index line not in the required format, dropped: {line.strip()[:120]}")
                continue
            if entry.pattern_id not in state.patterns:
                warnings.append(f"index names unknown pattern {entry.pattern_id!r}, dropped")
                continue
            entries.append(entry)
        listed = {e.pattern_id for e in entries}
        for pid in sorted(state.patterns):
            if pid not in listed:
                warnings.append(f"index omitted pattern {pid!r}; restored with its title")
                entries.append(IndexEntry(pid, state.patterns[pid].title))
        state.index = entries
        state.log.append(LogEntry(iteration=iteration, text=update.append_log.strip()))
        state.iteration = iteration
        return state

    def inherit_from(
        self, other: WikiState, *, source: str, evidence_meta: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    ) -> WikiState:
        """Return a copy of this wiki that also holds ``other``'s patterns it does not have yet (§3.13).

        Inherited patterns keep their bodies, index lines and evidence; every citation (retired ones included) is stamped
        ``meta.source_workspace`` (after the host's optional ``evidence_meta`` rewrite) because those trace ids belong
        to the source workspace and this workspace's trace store and invalidation feed cannot see them. Retired
        patterns are inherited too, so the maintainer does not recreate what the source already pruned. Impact history
        is not inherited: it records what THIS workspace validated. One log line records the transfer.
        """
        state = self.copy()
        added: list[str] = []
        for pid, pattern in other.patterns.items():
            if pid in state.patterns or pid in state.retired:
                continue
            copied = copy.deepcopy(pattern)
            copied.inherited_from = source
            copied.created_iteration = copied.updated_iteration = state.iteration
            for ref in [*copied.evidence, *copied.stale_evidence]:
                meta = evidence_meta(dict(ref.meta)) if evidence_meta else dict(ref.meta)
                ref.meta = {**meta, "source_workspace": source}
            state.patterns[pid] = copied
            added.append(pid)
        described = {e.pattern_id: e.description for e in other.index}
        listed = {e.pattern_id for e in state.index}
        for pid in added:
            if pid not in listed:
                state.index.append(IndexEntry(pid, described.get(pid, state.patterns[pid].title)))
        retired = [pid for pid in other.retired if pid not in state.retired and pid not in state.patterns]
        for pid in retired:
            copied_retired = copy.deepcopy(other.retired[pid])
            for ref in [*copied_retired.evidence, *copied_retired.stale_evidence]:
                meta = evidence_meta(dict(ref.meta)) if evidence_meta else dict(ref.meta)
                ref.meta = {**meta, "source_workspace": source}
            state.retired[pid] = copied_retired
        if added or retired:
            state.log.append(LogEntry(iteration=state.iteration, text=(
                f"[transfer] inherited {len(added)} pattern(s) and {len(retired)} retired pattern(s) from {source} "
                f"(wiki {other.digest[:12]})" + (": " + ", ".join(added) if added else ""))))
            state.meta.setdefault("inherited", []).append({"source": source, "wiki_digest": other.digest, "patterns": added, "retired": retired})
        return state

    def mark_stale(self, invalidated_trace_ids: set[str]) -> WikiState:
        """Move citations of invalidated traces to ``stale_evidence``. Nothing is deleted. Inherited citations carry
        ``meta.source_workspace`` and are never in this workspace's invalidation feed; the source owns their validity."""
        if not invalidated_trace_ids:
            return self
        state = self.copy()
        for pattern in state.patterns.values():
            still, stale = [], list(pattern.stale_evidence)
            for ref in pattern.evidence:
                (stale if ref.trace_id in invalidated_trace_ids else still).append(ref)
            pattern.evidence, pattern.stale_evidence = still, stale
        return state

    def apply_prune(
        self, update: PruneUpdate, *, iteration: int, bodies_shown: set[str], warnings: list[str] | None = None
    ) -> WikiState:
        """Return a new state with the pruner's merges and retirements applied (docs/paper-differences.md §3.6).

        Hard rules the model cannot override: a pattern cited by an accepted skill change is never retired or merged
        away; a merge target must exist and may only get a new body when the pruner was shown its current one;
        retired patterns move to ``retired`` with their evidence, they are not deleted; the index drops them.
        """
        state = self.copy()
        warnings = warnings if warnings is not None else []
        protected = state.protected_pattern_ids()
        removed: dict[str, tuple[str, str | None]] = {}  # id -> (reason, merged_into)

        def exists(pid: str, what: str) -> Pattern:
            if pid not in state.patterns:
                raise WikiError(f"{what} names unknown pattern {pid!r}; known: {sorted(state.patterns)}")
            return state.patterns[pid]

        for item in update.merges:
            if not isinstance(item, dict) or not isinstance(item.get("into"), str) or not isinstance(item.get("from"), list):
                raise WikiError("each merge needs 'into' (pattern id) and 'from' (list of pattern ids)")
            target_id = normalise_pattern_id(item["into"])
            if target_id in removed:
                raise WikiError(f"merge into {target_id!r}: that pattern is retired or merged away in this same update; merge into its survivor")
            target = exists(target_id, "merge")
            sources = [normalise_pattern_id(str(v)) for v in item["from"]]
            for sid in sources:
                if sid == target_id:
                    raise WikiError(f"merge into {target_id!r} lists itself as a source")
                if sid in protected:
                    raise WikiError(f"pattern {sid!r} is cited by an accepted skill and cannot be merged away")
                if sid in removed:
                    raise WikiError(f"pattern {sid!r} is retired or merged twice in one update")
            content = item.get("content")
            if content is not None:
                if target_id not in bodies_shown:
                    raise WikiError(f"merge into {target_id!r} rewrites a body you were not shown; omit 'content' or ask for the page")
                if not isinstance(content, str) or not content.strip():
                    raise WikiError(f"merge into {target_id!r}: 'content' must be a non-empty string when given")
                target.body = content
            if isinstance(item.get("title"), str) and item["title"].strip():
                target.title = item["title"].strip()
            for sid in sources:
                source = exists(sid, "merge")
                for ref in source.evidence:
                    if all(ref.trace_id != e.trace_id for e in target.evidence):
                        target.evidence.append(ref)
                for ref in source.stale_evidence:
                    if all(ref.trace_id != e.trace_id for e in target.stale_evidence):
                        target.stale_evidence.append(ref)
                removed[sid] = (str(item.get("reason") or f"merged into {target_id}"), target_id)
            target.updated_iteration = iteration
        for item in update.retire:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise WikiError("each retirement needs 'name' (pattern id) and 'reason'")
            pid = normalise_pattern_id(item["name"])
            exists(pid, "retire")
            if pid in protected:
                raise WikiError(f"pattern {pid!r} is cited by an accepted skill and cannot be retired")
            if pid in removed:
                raise WikiError(f"pattern {pid!r} is retired or merged twice in one update")
            removed[pid] = (str(item.get("reason") or "retired by the pruner"), None)
        for pid, (reason, merged_into) in removed.items():
            pattern = state.patterns.pop(pid)
            state.retired[pid] = RetiredPattern(
                id=pid, title=pattern.title, body=pattern.body, evidence=pattern.evidence, stale_evidence=pattern.stale_evidence,
                reason=reason, retired_iteration=iteration, merged_into=merged_into,
            )
        state.index = [e for e in state.index if e.pattern_id not in removed]
        summary = update.append_log.strip() or (
            f"Pruned: {len(removed)} pattern(s) retired or merged" if removed else "Prune pass: nothing to change"
        )
        state.log.append(LogEntry(iteration=iteration, text=f"[prune] {summary}"))
        return state

    def record_impact(self, entry: ImpactEntry) -> None:
        self.impact.append(entry)

    def evidence_trace_ids(self) -> set[str]:
        return {e.trace_id for p in self.patterns.values() for e in p.evidence}


def _title_from_body(body: str) -> str | None:
    for line in body.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return None


def _parse_evidence(values: Any, allowed: dict[str, dict[str, Any]], pid: str) -> list[EvidenceRef]:
    if not isinstance(values, list):
        raise WikiError(f"pattern {pid!r}: 'evidence' must be a list of trace ids or objects")
    refs: list[EvidenceRef] = []
    for value in values:
        if isinstance(value, str):
            ref = EvidenceRef(trace_id=value)
        elif isinstance(value, dict) and isinstance(value.get("trace_id"), str):
            ref = EvidenceRef(trace_id=value["trace_id"], note=str(value.get("note", "")))
        else:
            raise WikiError(f"pattern {pid!r}: evidence entry {value!r} must be a trace id or {{trace_id, note}}")
        if ref.trace_id not in allowed:
            raise WikiError(f"pattern {pid!r}: cites trace {ref.trace_id!r} which is not in this iteration's sample")
        ref.meta = copy.deepcopy(allowed.get(ref.trace_id) or {})
        refs.append(ref)
    return refs


__all__ = [
    "EvidenceRef",
    "ImpactEntry",
    "IndexEntry",
    "LogEntry",
    "MaintainerUpdate",
    "Pattern",
    "PatchOp",
    "PruneUpdate",
    "RetiredPattern",
    "WikiError",
    "WikiState",
]
