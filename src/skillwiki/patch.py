# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Patch operations shared by the Wiki Maintainer and the Skill Proposer (paper App. E).

Three operations on plain text: ``append``, ``replace`` and ``insert_after``.
``target`` must be an exact substring of the current text; ambiguity (more than
one occurrence) is an error because a patch that silently edits the wrong span
is worse than one that fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class PatchError(ValueError):
    """A patch operation could not be applied; the message is safe to show a model."""


@dataclass(frozen=True)
class PatchOp:
    op: str
    content: str
    target: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PatchOp:
        if not isinstance(value, dict) or "op" not in value:
            raise PatchError(f"patch operation must be an object with an 'op' key, got {value!r}")
        op = value["op"]
        if op not in {"append", "replace", "insert_after"}:
            raise PatchError(f"unknown patch op {op!r}; use append, replace or insert_after")
        content = value.get("content")
        if not isinstance(content, str):
            raise PatchError(f"patch op {op!r} needs a string 'content'")
        target = value.get("target")
        if op != "append" and (not isinstance(target, str) or not target):
            raise PatchError(f"patch op {op!r} needs a non-empty string 'target'")
        return cls(op=op, content=content, target=target if op != "append" else None)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"op": self.op, "content": self.content}
        if self.target is not None:
            value["target"] = self.target
        return value


def apply_patch(text: str, ops: list[PatchOp]) -> str:
    for op in ops:
        if op.op == "append":
            separator = "" if not text or text.endswith("\n") else "\n"
            text = text + separator + op.content
            continue
        assert op.target is not None
        count = text.count(op.target)
        if count == 0:
            raise PatchError(f"target not found: {op.target[:80]!r}")
        if count > 1:
            raise PatchError(f"target is ambiguous ({count} occurrences): {op.target[:80]!r}")
        if op.op == "replace":
            text = text.replace(op.target, op.content, 1)
        else:
            index = text.index(op.target) + len(op.target)
            separator = "" if op.content.startswith("\n") else "\n"
            text = text[:index] + separator + op.content + text[index:]
    return text


def parse_ops(values: Any) -> list[PatchOp]:
    if not isinstance(values, list) or not values:
        raise PatchError("edits must be a non-empty list of patch operations")
    return [PatchOp.from_dict(v) for v in values]
