# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Redaction of trace text before any role sees it (§3.10).

A host that learns from production traffic should not hand customer or reviewer identifiers to the model, nor persist
them in the raw layer. ``evolve(..., redact=...)`` applies a redactor to every collected trace's text before sampling
and before the trace store records it. Only the free text is redacted here: structured fields (``prediction``,
``truth``, ``meta``) are the host's own shapes and are the host's job to shape safely before it builds the trace.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Mapping

from .traces import Trace

Redactor = Callable[[str], str]


def regex_redactor(patterns: Mapping[str, str | re.Pattern[str]], *, replacement: str = "[REDACTED:{name}]") -> Redactor:
    """Build a redactor from named regular expressions; each match becomes ``replacement`` with ``{name}`` filled in.

    Patterns are applied in the given order, so put the most specific first. Names let the roles still tell one kind of
    redacted value from another (an email from an order number) without seeing either.
    """
    compiled = [(name, re.compile(p) if isinstance(p, str) else p) for name, p in patterns.items()]

    def redact(text: str) -> str:
        for name, pattern in compiled:
            text = pattern.sub(replacement.format(name=name), text)
        return text

    return redact


EMAIL = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
"""A ready pattern for e-mail addresses, the identifier most often left in reviewer notes."""


def redact_trace(trace: Trace, redactor: Redactor) -> Trace:
    """A copy of ``trace`` with redacted text; the original is left untouched (the host may still own it)."""
    text = redactor(trace.text)
    return trace if text == trace.text else dataclasses.replace(trace, text=text)


__all__ = ["EMAIL", "Redactor", "redact_trace", "regex_redactor"]
