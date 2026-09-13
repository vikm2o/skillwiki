# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Wiki Pruner (docs/paper-differences.md §3.6): keep the wiki small enough to be read.

The paper's wiki only grows. In production the maintainer's context eventually exceeds the model's window and the
loop stalls without an error. The pruner runs on a cadence or when the rendered wiki exceeds a budget, sees the
index plus one statistics line per pattern (bodies only within a separate budget, newest first), and proposes
merges of overlapping patterns and retirement of patterns that lost their evidence. Hard rules live in
:meth:`skillwiki.wiki.WikiState.apply_prune`, not in the prompt.
"""

from __future__ import annotations

from ..config import EvolveConfig
from ..model import ChatModel, ModelRequest, TextPart
from ..tools import _json_objects
from ..wiki import PruneUpdate, WikiError, WikiState

SYSTEM_PROMPT = """You are the Wiki Pruner for a skill-evolution system.

The wiki catalogues patterns (root causes, workarounds) observed while an agent executed tasks. It is read in full by
the Wiki Maintainer every iteration, so it must stay concise. Your job is consolidation, not authorship: merge
patterns that describe the same root cause, and retire patterns whose evidence is gone or that were never useful.
Everything you retire stays readable in wiki/retired.md; nothing is deleted.

## Your input
1. The pattern index
2. One statistics line per pattern: size, live and stale evidence counts, when it was created and last updated,
   and whether an ACCEPTED skill cites it (those are protected: you cannot retire or merge them away)
3. The full bodies of some patterns (most recently updated first, within a budget). You may rewrite the body of a
   merge target ONLY if its body is shown to you.

## Your output: ONE JSON object
{
  "merges": [{"into": "<surviving pattern>", "from": ["<absorbed pattern>", ...], "title": "<optional new title>",
              "content": "<optional new body; only when the target body was shown>", "reason": "<why>"}],
  "retire": [{"name": "<pattern>", "reason": "<why>"}],
  "append_log": "one or two sentences summarising what changed and why"
}
Empty lists are a valid answer when nothing should change.

## Rules
1. Merge only patterns with the SAME root cause; different symptoms of one cause belong together, different causes
   do not. The absorbed patterns' evidence is carried onto the target automatically.
2. Retire a pattern when all its evidence is stale, or when it has not been updated for many iterations and no skill
   ever cited it. Prefer merging over retiring when the content is still useful.
3. Never retire or merge away a protected pattern.
4. Do not invent new patterns. Do not change bodies you were not shown.
5. Return only the JSON object."""


class WikiPruner:
    def __init__(self, model: ChatModel, config: EvolveConfig):
        self.model, self.config = model, config

    def _context(self, wiki: WikiState, iteration: int) -> tuple[str, set[str]]:
        protected = wiki.protected_pattern_ids()
        lines = [f"# Prune pass at iteration {iteration}", "", "## Index", wiki.render_index(), "## Pattern statistics"]
        for pid, pattern in sorted(wiki.patterns.items()):
            lines.append(
                f"- {pid}: {len(pattern.body)} chars, {len(pattern.evidence)} live / {len(pattern.stale_evidence)} stale evidence, "
                f"created {pattern.created_iteration}, updated {pattern.updated_iteration}"
                + (", PROTECTED (cited by an accepted skill)" if pid in protected else "")
            )
        shown: set[str] = set()
        budget = self.config.prune_bodies_char_cap
        bodies: list[str] = []
        for pid, pattern in sorted(wiki.patterns.items(), key=lambda kv: (-kv[1].updated_iteration, kv[0])):
            page = pattern.to_markdown()
            if len(page) > budget:
                continue
            bodies.append(f"<<< wiki/patterns/{pid}.md >>>\n{page}")
            shown.add(pid)
            budget -= len(page)
        lines += ["", f"## Pattern bodies shown ({len(shown)} of {len(wiki.patterns)})", *bodies]
        if wiki.retired:
            lines += ["", "## Already retired (for reference)", *(f"- {pid}: {r.reason}" for pid, r in sorted(wiki.retired.items()))]
        return "\n".join(lines), shown

    async def prune(self, wiki: WikiState, *, iteration: int) -> tuple[WikiState, list[str]]:
        """Return (pruned wiki, warnings). Unusable output is fed back once; a second failure leaves the wiki unchanged."""
        warnings: list[str] = []
        text, shown = self._context(wiki, iteration)
        parts = [TextPart(text)]
        for attempt in range(self.config.role_retries + 1):
            response = await self.model.complete(
                ModelRequest(role="wiki_pruner", system=SYSTEM_PROMPT, parts=parts, max_tokens=self.config.max_tokens)
            )
            try:
                objects = _json_objects(response.text)
                if not objects:
                    raise WikiError("no JSON object found in the response")
                update = PruneUpdate.from_dict(objects[0])
                return wiki.apply_prune(update, iteration=iteration, bodies_shown=shown, warnings=warnings), warnings
            except WikiError as exc:
                warnings.append(f"pruner output rejected (attempt {attempt + 1}): {exc}")
                parts = [*parts, TextPart(f"\n## Your previous output was rejected\n{exc}\nReturn the corrected JSON object only.")]
        warnings.append(f"wiki not pruned at iteration {iteration}: pruner output could not be applied")
        return wiki, warnings


__all__ = ["SYSTEM_PROMPT", "WikiPruner"]
