# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""The Skill Layer: skill documents, the layered active skill set, and atomic proposals.

A skill is ``SKILL.md`` (frontmatter + procedural body) plus ``PURPOSE.md``
(why it exists: origin, patterns addressed, evolution history). Hosts with
structured skills convert to and from :class:`SkillDocument` with their own
codec; the module only ever patches the markdown body.

Skills live in ordered **layers** (docs/paper-differences.md §3.2), outer to
inner, e.g. ``global`` → ``tenant`` → ``project``. The proposer may create,
patch and retire only in editable layers; a skill in a non-editable outer
layer can be *overridden* by an inner skill that names it. A single unnamed
layer is the paper's flat ``skills/`` directory and renders identically.
"""

from __future__ import annotations

import copy
import difflib
import re
from dataclasses import dataclass, field
from typing import Any

from . import frontmatter
from .documents import SCHEMA_VERSION, digest
from .patch import PatchError, PatchOp, apply_patch, parse_ops

SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9\-_]{0,79}$")
DEFAULT_LAYER = "local"
ACTIONS = {"create", "patch", "retire", "override", "promote", "no_action"}


class ProposalError(ValueError):
    """A proposal could not be applied; the message is safe to show a model."""


@dataclass(frozen=True)
class Layer:
    """One skill layer. ``editable`` is decided by the host: usually only the innermost layer is."""

    name: str
    editable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "editable": self.editable}


@dataclass
class SkillDocument:
    name: str
    body: str
    purpose: str = ""
    frontmatter: dict[str, Any] = field(default_factory=dict)
    protected: bool = False
    layer: str = DEFAULT_LAYER
    overrides: str | None = None  # name of an outer-layer skill this one replaces for readers of this layer

    @property
    def description(self) -> str:
        return str(self.frontmatter.get("description", ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "body": self.body,
            "purpose": self.purpose,
            "frontmatter": self.frontmatter,
            "protected": self.protected,
            "layer": self.layer,
            "overrides": self.overrides,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SkillDocument:
        return cls(
            name=value["name"],
            body=value.get("body", ""),
            purpose=value.get("purpose", ""),
            frontmatter=dict(value.get("frontmatter", {})),
            protected=bool(value.get("protected", False)),
            layer=str(value.get("layer") or DEFAULT_LAYER),
            overrides=value.get("overrides") or None,
        )

    _FILE_FIELDS = ("protected", "layer", "overrides")  # rendered into SKILL.md so the file round-trips; not part of content_digest

    def render_skill_md(self, *, file_fields: bool = True) -> str:
        """SKILL.md text. ``file_fields=False`` omits ``protected``/``layer``/``overrides``: loop bookkeeping the
        inference agent has no use for (see :meth:`SkillSet.render_for_prompt` with ``effective=True``)."""
        fields = {"name": self.name, **{k: v for k, v in self.frontmatter.items() if k not in ("name", *self._FILE_FIELDS)}}
        if file_fields:
            if self.protected:
                fields["protected"] = True
            if self.layer != DEFAULT_LAYER:
                fields["layer"] = self.layer
            if self.overrides:
                fields["overrides"] = self.overrides
        return frontmatter.render(fields) + self.body

    @classmethod
    def parse_skill_md(
        cls,
        text: str,
        *,
        purpose: str = "",
        protected: bool = False,
        fallback_name: str = "",
        layer: str = DEFAULT_LAYER,
        trusted: bool = False,
    ) -> SkillDocument:
        """Parse a SKILL.md. ``protected``, ``layer`` and ``overrides`` in the frontmatter are honoured only when
        ``trusted`` (a host reading its own ``render_skill_md`` output); model-authored text never sets them, the
        keyword arguments do. Untrusted values are dropped rather than kept in ``frontmatter`` so they cannot
        enter ``content_digest``."""
        try:
            fields, body = frontmatter.split(text)
        except ValueError as exc:  # a YAML list or nested block: this format is one 'key: value' per line
            raise ProposalError(
                f"SKILL.md frontmatter: {exc}. Use one 'key: value' line per field; lists and nested YAML are not supported"
            ) from exc
        for key in fields:
            if not frontmatter.KEY.match(key):  # a key render() could not write back would crash the diff later
                raise ProposalError(f"SKILL.md frontmatter key {key!r} is not allowed: use letters, digits, '_' or '-' (e.g. when_to_use)")
        name = str(fields.pop("name", fallback_name)).strip()
        if not SKILL_NAME.match(name):
            raise ProposalError(f"invalid skill name {name!r}: use lowercase letters, digits, '-' or '_'")
        file_fields = {k: fields.pop(k) for k in cls._FILE_FIELDS if k in fields}
        overrides = None
        if trusted:
            protected = bool(file_fields.get("protected", protected))
            layer = str(file_fields.get("layer", layer))
            overrides = file_fields.get("overrides") or None
        return cls(name=name, body=body, purpose=purpose, frontmatter=fields, protected=protected, layer=layer, overrides=overrides)

    def to_files(self, *, nested: bool = False) -> dict[str, str]:
        root = f"skills/{self.layer}/{self.name}" if nested else f"skills/{self.name}"
        return {f"{root}/SKILL.md": self.render_skill_md(), f"{root}/PURPOSE.md": self.purpose}

    def render_for_prompt(self, *, note: str = "", file_fields: bool = True) -> str:
        head = f"### Skill: {self.name}" + (f" {note}" if note else "")
        return f"{head}\n{self.render_skill_md(file_fields=file_fields).rstrip()}\n"


@dataclass
class SkillSet:
    skills: dict[str, SkillDocument] = field(default_factory=dict)
    layers: list[Layer] = field(default_factory=lambda: [Layer(DEFAULT_LAYER, True)])  # outer -> inner

    @classmethod
    def from_documents(cls, documents: list[SkillDocument], *, layers: list[Layer] | None = None) -> SkillSet:
        result = cls()
        if layers:
            result.layers = list(layers)
        for document in documents:
            if document.name in result.skills:
                raise ProposalError(f"duplicate skill name {document.name!r}")
            result.skills[document.name] = document
        seen = {layer.name for layer in result.layers}
        for document in documents:  # a document in an undeclared layer gets that layer appended (inner, editable)
            if document.layer not in seen:
                result.layers.append(Layer(document.layer, True))
                seen.add(document.layer)
        return result

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "layers": [layer.to_dict() for layer in self.layers],
            "skills": [self.skills[name].to_dict() for name in sorted(self.skills)],
        }

    @classmethod
    def from_document(cls, value: dict[str, Any]) -> SkillSet:
        layers = [Layer(str(v["name"]), bool(v.get("editable", True))) for v in value.get("layers", [])] or None
        return cls.from_documents([SkillDocument.from_dict(v) for v in value.get("skills", [])], layers=layers)

    @property
    def digest(self) -> str:
        return digest(self.to_document())

    def content_digest(self, *, ignore_frontmatter: tuple[str, ...] = ("id",)) -> str:
        """Digest of what the agent actually reads: names, bodies, frontmatter (minus host identity fields) and, when
        set, ``overrides`` (it hides an outer skill from the agent, so two candidates that differ only there read
        differently). Excludes PURPOSE.md and the layer, so the same proposal made twice has the same digest even
        though ``_append_history`` stamps the iteration into purpose."""
        return digest(
            [
                {
                    "name": s.name,
                    "body": s.body,
                    "frontmatter": {k: v for k, v in s.frontmatter.items() if k not in ignore_frontmatter},
                    **({"overrides": s.overrides} if s.overrides else {}),
                }
                for s in (self.skills[n] for n in sorted(self.skills))
            ]
        )

    def copy(self) -> SkillSet:
        return copy.deepcopy(self)

    # -- layers ----------------------------------------------------------------------------------------------------

    @property
    def multi_layer(self) -> bool:
        return len(self.layers) > 1

    def layer(self, name: str) -> Layer:
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise ProposalError(f"unknown layer {name!r}; layers: {[layer.name for layer in self.layers]}")

    def layer_index(self, name: str) -> int:
        return [layer.name for layer in self.layers].index(self.layer(name).name)

    @property
    def editable_layers(self) -> list[Layer]:
        return [layer for layer in self.layers if layer.editable]

    @property
    def innermost_editable(self) -> Layer:
        editable = self.editable_layers
        if not editable:
            raise ProposalError("no editable layer: the host made every layer read-only")
        return editable[-1]

    def overridden(self) -> dict[str, str]:
        """``{outer skill name: inner skill name}`` for every override whose target exists."""
        return {s.overrides: s.name for s in self.skills.values() if s.overrides and s.overrides in self.skills}

    def effective(self) -> list[SkillDocument]:
        """What a reader of the innermost layer sees: every skill except outer ones hidden by an override."""
        hidden = set(self.overridden())
        return [self.skills[n] for n in sorted(self.skills) if n not in hidden]

    def to_files(self) -> dict[str, str]:
        files: dict[str, str] = {}
        for name in sorted(self.skills):
            files.update(self.skills[name].to_files(nested=self.multi_layer))
        return files

    def render_for_prompt(self, *, effective: bool = False) -> str:
        """Markdown of the skill set for a prompt.

        The default is the ROLES' view: every skill, grouped by layer, with ``(protected)``, ``(overrides X)`` and
        ``(overridden by X)`` notes, so the maintainer and proposer see what is hidden and why. ``effective=True``
        is the AGENT's view: only :meth:`effective` skills, no bookkeeping notes. Hosts that build the inference
        prompt from this method should pass ``effective=True``; an overridden skill must not reach the agent.
        """
        if not self.skills:
            return "(no active skills)\n"
        if effective:
            visible = self.effective()
            return "\n".join(s.render_for_prompt(file_fields=False) for s in visible) if visible else "(no active skills)\n"
        if not self.multi_layer:
            return "\n".join(self.skills[name].render_for_prompt() for name in sorted(self.skills))
        overridden = self.overridden()
        blocks: list[str] = []
        for layer in self.layers:
            members = [self.skills[n] for n in sorted(self.skills) if self.skills[n].layer == layer.name]
            if not members:
                continue
            blocks.append(f"## Layer: {layer.name} ({'editable' if layer.editable else 'read-only'})")
            for skill in members:
                notes = []
                if skill.protected:
                    notes.append("(protected)")
                if skill.name in overridden:
                    notes.append(f"(overridden by {overridden[skill.name]})")
                if skill.overrides:
                    notes.append(f"(overrides {skill.overrides})")
                blocks.append(skill.render_for_prompt(note=" ".join(notes)))
        return "\n".join(blocks)

    def __len__(self) -> int:
        return len(self.skills)


@dataclass
class Proposal:
    """One atomic skill change (paper §3.2.3): create, patch, retire, override or promote exactly one skill, or no_action.

    A merge is a ``create`` whose ``supersedes`` names the skills it replaces; the replacement and the retirements
    are one proposal, so a skill set never observes half a merge. ``override`` is a ``create`` in the innermost
    editable layer that shadows a read-only outer skill (``name`` is the new skill, ``overrides`` the outer one).
    ``promote`` moves an editable skill one layer outward; hosts allow it with ``EvolveConfig.allow_promotions``.
    """

    action: str
    name: str | None = None
    skill_md: str | None = None
    purpose_md: str | None = None
    edits: list[PatchOp] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    overrides: str | None = None
    rationale: str = ""
    citations: list[str] = field(default_factory=list)  # pattern ids and/or trace ids the proposer relied on

    @classmethod
    def from_dict(cls, value: Any, *, allow_promotions: bool = False) -> Proposal:
        if not isinstance(value, dict) or "action" not in value:
            raise ProposalError("proposal must be a JSON object with an 'action' key")
        action = value["action"]
        allowed = ACTIONS - ({"promote"} if not allow_promotions else set())
        if action not in allowed:
            raise ProposalError(f"unknown action {action!r}; use one of {sorted(allowed)}")
        proposal = cls(
            action=action,
            name=value.get("name"),
            skill_md=value.get("skill_md"),
            purpose_md=value.get("purpose_md"),
            supersedes=[str(v) for v in value.get("supersedes", []) or []],
            overrides=str(value["overrides"]) if value.get("overrides") else None,
            rationale=str(value.get("rationale", "") or ""),
            citations=[str(v) for v in value.get("citations", []) or []],
        )
        if action == "no_action":
            return proposal
        if not isinstance(proposal.name, str) or not SKILL_NAME.match(proposal.name):
            raise ProposalError(f"'name' must be a snake_case/kebab-case skill directory name, got {proposal.name!r}")
        if action in {"create", "override"}:
            if not isinstance(proposal.skill_md, str) or not proposal.skill_md.strip():
                raise ProposalError(f"{action} needs 'skill_md' with YAML-style frontmatter and the full instructions")
            if not isinstance(proposal.purpose_md, str) or not proposal.purpose_md.strip():
                raise ProposalError(f"{action} needs 'purpose_md' (Origin, Patterns Addressed, Evolution History)")
        if action == "override" and not proposal.overrides:
            raise ProposalError("override needs 'overrides': the name of the read-only skill it replaces")
        if action != "override" and proposal.overrides:
            raise ProposalError("'overrides' is only valid with action 'override'")
        if action == "patch":
            try:
                proposal.edits = parse_ops(value.get("edits"))
            except PatchError as exc:
                raise ProposalError(str(exc)) from exc
        if len(set(proposal.supersedes)) != len(proposal.supersedes):
            raise ProposalError("'supersedes' must not repeat a skill")
        if proposal.supersedes and action != "create":
            raise ProposalError("'supersedes' is only valid with action 'create' (a merge)")
        return proposal

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "name": self.name,
            "skill_md": self.skill_md,
            "purpose_md": self.purpose_md,
            "edits": [op.to_dict() for op in self.edits],
            "supersedes": list(self.supersedes),
            "overrides": self.overrides,
            "rationale": self.rationale,
            "citations": list(self.citations),
        }

    @property
    def is_merge(self) -> bool:
        return self.action == "create" and bool(self.supersedes)

    def rendered_body(self) -> str | None:
        """Full content of the proposed change, retained in skill-impact for rejected proposals."""
        if self.action in {"create", "override"}:
            return f"{self.skill_md}\n\n--- PURPOSE.md ---\n{self.purpose_md}"
        if self.action == "patch":
            return "\n".join(f"{op.op}: target={op.target!r}\n{op.content}" for op in self.edits)
        return None


def apply_proposal(skill_set: SkillSet, proposal: Proposal, *, iteration: int, allow_protected: bool = False) -> SkillSet:
    """Return the candidate skill set S'_k = Apply(S_{k-1}, P_k). Raises :class:`ProposalError` on any inconsistency."""
    candidate = skill_set.copy()
    if proposal.action == "no_action":
        return candidate
    assert proposal.name is not None
    target_layer = candidate.innermost_editable

    def existing(name: str) -> SkillDocument:
        if name not in candidate.skills:
            raise ProposalError(f"skill {name!r} does not exist; existing skills: {sorted(candidate.skills) or 'none'}")
        return candidate.skills[name]

    def guard(name: str) -> SkillDocument:
        """A skill the proposer may change or retire: exists, not protected, in an editable layer, not already overridden."""
        skill = existing(name)
        if skill.protected and not allow_protected:
            raise ProposalError(f"skill {name!r} is protected by the host and cannot be changed or retired")
        if not candidate.layer(skill.layer).editable:
            raise ProposalError(
                f"skill {name!r} is in read-only layer {skill.layer!r}; use action 'override' to shadow it with a "
                f"{target_layer.name!r} skill instead of editing it"
            )
        overridden = candidate.overridden()
        if name in overridden:
            raise ProposalError(f"skill {name!r} is already overridden by {overridden[name]!r}; change that skill instead")
        return skill

    if proposal.action == "retire":
        guard(proposal.name)
        del candidate.skills[proposal.name]
        return candidate

    if proposal.action == "patch":
        skill = guard(proposal.name)
        try:
            skill.body = apply_patch(skill.body, proposal.edits)
        except PatchError as exc:
            raise ProposalError(f"skill {proposal.name!r}: {exc}") from exc
        skill.purpose = _append_history(skill.purpose, iteration, f"Patched: {proposal.rationale or 'no rationale given'}")
        return candidate

    if proposal.action == "promote":
        skill = guard(proposal.name)
        index = candidate.layer_index(skill.layer)
        if index == 0:
            raise ProposalError(f"skill {proposal.name!r} is already in the outermost layer {skill.layer!r}")
        skill.layer = candidate.layers[index - 1].name
        skill.purpose = _append_history(skill.purpose, iteration, f"Promoted to layer {skill.layer}: {proposal.rationale or 'no rationale'}")
        return candidate

    if proposal.action == "override":
        assert proposal.overrides is not None
        outer = existing(proposal.overrides)
        if outer.protected and not allow_protected:
            raise ProposalError(f"skill {proposal.overrides!r} is protected by the host and cannot be overridden")
        if candidate.layer(outer.layer).editable:
            raise ProposalError(f"skill {proposal.overrides!r} is editable; use action 'patch' rather than 'override'")
        already = candidate.overridden()
        if proposal.overrides in already:
            raise ProposalError(f"skill {proposal.overrides!r} is already overridden by {already[proposal.overrides]!r}")
        if proposal.name in candidate.skills:
            raise ProposalError(f"skill {proposal.name!r} already exists; choose a new name for the override")
        assert proposal.skill_md is not None
        document = SkillDocument.parse_skill_md(
            proposal.skill_md, purpose=proposal.purpose_md or "", fallback_name=proposal.name, layer=target_layer.name
        )
        if document.name != proposal.name:
            raise ProposalError(f"frontmatter name {document.name!r} does not match proposal name {proposal.name!r}")
        document.overrides = proposal.overrides
        document.purpose = _append_history(document.purpose, iteration, f"Overrides {proposal.overrides} ({outer.layer})")
        candidate.skills[document.name] = document
        return candidate

    # create (optionally a merge)
    for name in proposal.supersedes:
        guard(name)
    if proposal.name in candidate.skills and proposal.name not in proposal.supersedes:
        raise ProposalError(f"skill {proposal.name!r} already exists; use action 'patch' or list it in 'supersedes'")
    assert proposal.skill_md is not None
    document = SkillDocument.parse_skill_md(
        proposal.skill_md, purpose=proposal.purpose_md or "", fallback_name=proposal.name, layer=target_layer.name
    )
    if document.name != proposal.name:
        raise ProposalError(f"frontmatter name {document.name!r} does not match proposal name {proposal.name!r}")
    for name in proposal.supersedes:
        del candidate.skills[name]
    if proposal.supersedes:
        document.purpose = _append_history(document.purpose, iteration, f"Created by merging: {', '.join(proposal.supersedes)}")
    candidate.skills[document.name] = document
    return candidate


def _append_history(purpose: str, iteration: int, line: str) -> str:
    marker = "## Evolution History"
    entry = f"- Iteration {iteration}: {line}"
    if marker in purpose:
        return purpose.rstrip() + "\n" + entry + "\n"
    return purpose.rstrip() + ("\n\n" if purpose.strip() else "") + marker + "\n" + entry + "\n"


def unified_diff(before: SkillSet, after: SkillSet) -> str:
    """Unified diff over the rendered ``skills/`` files, as recorded in skill-impact.md (paper §3.2.4)."""
    old, new = before.to_files(), after.to_files()
    chunks: list[str] = []
    for path in sorted(set(old) | set(new)):
        a, b = old.get(path), new.get(path)
        if a == b:
            continue
        lines = difflib.unified_diff(
            (a or "").splitlines(keepends=True),
            (b or "").splitlines(keepends=True),
            fromfile=path if a is not None else "/dev/null",
            tofile=path if b is not None else "/dev/null",
        )
        chunks.append("".join(lines))
    return "\n".join(chunks)


__all__ = ["ACTIONS", "DEFAULT_LAYER", "Layer", "Proposal", "ProposalError", "SkillDocument", "SkillSet", "apply_proposal", "unified_diff"]
