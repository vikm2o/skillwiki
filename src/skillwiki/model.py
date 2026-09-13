# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Provider-agnostic chat model protocol.

A host implements :class:`ChatModel` once. Every role call is a single request
carrying a system prompt and one user message made of text and image parts, and
expects plain text back. Tool use is carried inside the text (see
:mod:`skillwiki.tools`), so providers without native tool calling, and metered
transports that forbid it, work unchanged. A model that sets ``supports_tools``
receives the same tools as :class:`ToolSpec` entries on the request and answers
with :class:`ToolInvocation` entries instead (§3.12); the harness treats both
paths identically.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    """A base64-encoded image. ``ref`` names it in text placeholders once it is no longer attached."""

    media_type: str
    data_base64: str
    ref: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"media_type": self.media_type, "data_base64": self.data_base64, "ref": self.ref}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ImagePart:
        return cls(media_type=value["media_type"], data_base64=value["data_base64"], ref=value.get("ref", ""))


Part = TextPart | ImagePart


@dataclass(frozen=True)
class ToolSpec:
    """A tool offered to a natively tool-calling model; ``parameters`` is a JSON schema object."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolInvocation:
    """A tool call the model made natively (as opposed to inside its text)."""

    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ModelRequest:
    role: str
    system: str
    parts: Sequence[Part]
    max_tokens: int = 4096
    tools: Sequence[ToolSpec] = ()  # only set for models whose ``supports_tools`` is true

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.parts if isinstance(p, TextPart))

    @property
    def images(self) -> list[ImagePart]:
        return [p for p in self.parts if isinstance(p, ImagePart)]


@dataclass(frozen=True)
class ModelResponse:
    text: str
    usage: dict[str, Any] = field(default_factory=dict)
    tool_calls: Sequence[ToolInvocation] = ()  # filled by natively tool-calling models when the request offered tools


class ChatModel(Protocol):
    """``supports_tools`` is an optional attribute; absent or false means the text protocol is used."""

    async def complete(self, request: ModelRequest) -> ModelResponse: ...


class ScriptedChatModel:
    """Deterministic model for tests and examples.

    ``script`` is either a list of responses consumed in order, or a callable
    that receives the request and returns the response text. Every request is
    recorded in ``calls``.
    """

    def __init__(self, script: Sequence[str] | Callable[[ModelRequest], str | Awaitable[str]], *, native_tools: bool = False):
        self._script = list(script) if not callable(script) else script
        self.calls: list[ModelRequest] = []
        self.supports_tools = native_tools  # when set, a scripted tool-call JSON is delivered as a native tool call instead of text

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        if callable(self._script):
            result = self._script(request)
            if hasattr(result, "__await__"):
                result = await result
            text = str(result)
        elif not self._script:
            raise RuntimeError("ScriptedChatModel ran out of scripted responses")
        else:
            text = self._script.pop(0)
        if self.supports_tools and request.tools:
            from .tools import ToolCallError, parse_tool_call  # local: tools imports this module

            try:
                call = parse_tool_call(text, allowed={t.name for t in request.tools})
            except ToolCallError:
                return ModelResponse(text=text)  # the script answered in prose; the proposer sees no tool call, as a real model might
            return ModelResponse(text="", tool_calls=(ToolInvocation(call.name, dict(call.args)),))
        return ModelResponse(text=text)
