# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Structured-text tool protocol for the ReAct proposer.

The model emits exactly one tool call per turn as a JSON object in its text,
``{"tool": "<name>", "args": {...}}``. The harness executes it, appends the
observation to a scratchpad and sends the whole scratchpad again as ONE user
message on the next turn. This works on any provider, including metered
transports that forbid native tool calling or multi-message conversations.

Media returned by a tool are attached only in the turn that fetched them; the
scratchpad keeps a text placeholder afterwards, so cost does not grow with the
square of the number of turns.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .model import ImagePart, Part, TextPart

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class ToolCallError(ValueError):
    """The model's text did not contain a usable tool call; the message is safe to show a model."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]

    def render(self) -> str:
        return json.dumps({"tool": self.name, "args": self.args}, ensure_ascii=False, sort_keys=True)


def _json_objects(text: str) -> list[dict[str, Any]]:
    """Every top-level JSON object in ``text``, in order (fenced blocks first, then bare scans)."""
    found: list[dict[str, Any]] = []
    candidates = [m.group(1) for m in _FENCE.finditer(text)] or [text]
    for chunk in candidates:
        decoder = json.JSONDecoder()
        index = 0
        while True:
            start = chunk.find("{", index)
            if start < 0:
                break
            try:
                value, end = decoder.raw_decode(chunk, start)
            except ValueError:
                index = start + 1
                continue
            if isinstance(value, dict):
                found.append(value)
            index = end
    return found


def parse_tool_call(text: str, *, allowed: set[str] | None = None) -> ToolCall:
    calls = [v for v in _json_objects(text) if isinstance(v.get("tool"), str)]
    if not calls:
        raise ToolCallError(
            "no tool call found. Reply with exactly one JSON object of the form "
            '{"tool": "<name>", "args": {...}} and nothing else that looks like JSON.'
        )
    if len(calls) > 1:
        raise ToolCallError(f"found {len(calls)} tool calls; make exactly one call per turn")
    value = calls[0]
    name = value["tool"]
    args = value.get("args")
    if args is None:
        args = {k: v for k, v in value.items() if k != "tool"}
    if not isinstance(args, dict):
        raise ToolCallError("'args' must be a JSON object")
    if allowed is not None and name not in allowed:
        raise ToolCallError(f"unknown tool {name!r}; available tools: {sorted(allowed)}")
    return ToolCall(name=name, args=args)


@dataclass
class Turn:
    call: ToolCall | None
    model_text: str
    observation: str
    media: list[ImagePart] = field(default_factory=list)


@dataclass
class Scratchpad:
    """The folded ReAct transcript. ``render`` truncates the oldest observations first when over budget."""

    turns: list[Turn] = field(default_factory=list)

    def add(self, call: ToolCall | None, model_text: str, observation: str, media: list[ImagePart] | None = None) -> None:
        self.turns.append(Turn(call=call, model_text=model_text, observation=observation, media=list(media or [])))

    def render(self, *, char_cap: int) -> str:
        blocks: list[str] = []
        for i, turn in enumerate(self.turns, start=1):
            head = f"### Turn {i}\nYou called: {turn.call.render() if turn.call else '(no valid tool call)'}"
            media_note = ""
            if turn.media:
                shown = "attached below" if i == len(self.turns) else "shown at that turn, no longer attached"
                refs = ", ".join(m.ref or f"image {j + 1}" for j, m in enumerate(turn.media))
                media_note = f"\n[{len(turn.media)} image(s): {refs} -- {shown}]"
            blocks.append(f"{head}\nObservation:\n{turn.observation}{media_note}")
        text = "\n\n".join(blocks)
        if len(text) <= char_cap:
            return text
        # Fold: keep the latest turns whole, summarise older observations to their first line.
        folded = list(blocks)
        for i in range(len(folded) - 1):
            first_line = self.turns[i].observation.strip().splitlines()[:1]
            folded[i] = (
                f"### Turn {i + 1}\nYou called: {self.turns[i].call.render() if self.turns[i].call else '(invalid)'}\n"
                f"Observation (folded): {first_line[0] if first_line else ''} [...]"
            )
            text = "\n\n".join(folded)
            if len(text) <= char_cap:
                break
        return text if len(text) <= char_cap else text[-char_cap:]

    def parts(self, *, char_cap: int) -> list[Part]:
        """Text of the folded transcript plus only the most recent turn's media."""
        parts: list[Part] = [TextPart(self.render(char_cap=char_cap))]
        if self.turns and self.turns[-1].media:
            parts.extend(self.turns[-1].media)
        return parts


__all__ = ["Scratchpad", "ToolCall", "ToolCallError", "Turn", "parse_tool_call"]
