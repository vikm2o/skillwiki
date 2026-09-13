# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""A tiny, round-trippable frontmatter format used by the markdown renderers.

Each line is ``key: value``. A value is rendered bare when it is a simple string
that cannot be mistaken for JSON; otherwise it is JSON-encoded. Parsing tries
JSON first and falls back to the raw string, so ``render(parse(x)) == x`` for
anything this module produced.
"""

from __future__ import annotations

import json
import re
from typing import Any

_SAFE_BARE = re.compile(r"^[A-Za-z][A-Za-z0-9 _./,'()\-]*$")
_JSON_LIKE = re.compile(r"^(true|false|null|-?\d)")
KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]*$")
"""The keys :func:`render` can write. :func:`split` accepts any key so a caller can report a bad one itself."""


def render_value(value: Any) -> str:
    if isinstance(value, str) and _SAFE_BARE.match(value) and not _JSON_LIKE.match(value) and value.strip() == value:
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def parse_value(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        return text


def render(fields: dict[str, Any]) -> str:
    lines = ["---"]
    for key in fields:
        if not KEY.match(key):
            raise ValueError(f"frontmatter key not representable: {key!r}")
        lines.append(f"{key}: {render_value(fields[key])}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def split(text: str) -> tuple[dict[str, Any], str]:
    """Return (frontmatter fields, body). A document without frontmatter yields ({}, text). CRLF line endings are
    normalised to LF first, so a file written on Windows parses like one written here."""
    text = text.replace("\r\n", "\n")
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        if text.rstrip("\n").endswith("\n---"):
            end = len(text.rstrip("\n")) - 4
            body = ""
        else:
            return {}, text
    else:
        body = text[end + 5 :]
    fields: dict[str, Any] = {}
    for line in text[4:end].splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"malformed frontmatter line: {line!r}")
        fields[key.strip()] = parse_value(value)
    return fields, body
