# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""Skill Proposer (paper §3.2.3, App. E.3): a ReAct agent that reads the wiki and traces on demand, then
proposes ONE atomic skill change."""

from __future__ import annotations

import json
from typing import Any

from ..config import EvolveConfig
from ..model import ChatModel, ImagePart, ModelRequest, Part, TextPart, ToolSpec
from ..skills import Proposal, ProposalError, SkillSet, apply_proposal
from ..tools import Scratchpad, ToolCall, ToolCallError, parse_tool_call
from ..traces import Trace, summarize_outcomes, summarize_skill_usage
from ..wiki import WikiState, normalise_pattern_id

TOOLS = {"read_index", "read_pattern", "read_log", "read_impact", "list_traces", "read_trace", "read_skill", "finish"}

_STRING = {"type": "string"}
TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec("read_index", "wiki/index.md, the catalogue of known patterns. Start here.", {"type": "object", "properties": {}}),
    ToolSpec("read_impact", "wiki/skill-impact.md: everything tried before with outcomes, including rejected proposals.", {"type": "object", "properties": {}}),
    ToolSpec("read_log", "The chronological evolution log.", {"type": "object", "properties": {}}),
    ToolSpec("read_pattern", "One pattern page with root cause, evidence and workarounds.",
             {"type": "object", "properties": {"name": _STRING}, "required": ["name"]}),
    ToolSpec("list_traces", "Every training task with PASS/FAIL, prediction and truth.",
             {"type": "object", "properties": {"status": {"type": "string", "enum": ["failing", "passing", "all"]}}}),
    ToolSpec("read_trace", "The full execution trace of one task; include_media only when the images are needed.",
             {"type": "object", "properties": {"id": _STRING, "include_media": {"type": "boolean"}}, "required": ["id"]}),
    ToolSpec("read_skill", "The current SKILL.md and PURPOSE.md of an active skill.",
             {"type": "object", "properties": {"name": _STRING}, "required": ["name"]}),
    ToolSpec("finish", "Submit the proposal object described in the instructions.",
             {"type": "object", "properties": {"proposal": {"type": "object"}}, "required": ["proposal"]}),
)
"""The same eight tools as JSON schemas, offered to models that call tools natively (§3.12)."""

TEXT_PROTOCOL = 'Reply with EXACTLY ONE JSON object per turn: {"tool": "<name>", "args": {...}}. Nothing else that looks like JSON.'
NATIVE_PROTOCOL = "Call EXACTLY ONE of the provided tools per turn and put nothing else in the reply."

SYSTEM_PROMPT = """You are the Skill Proposer for an agent that solves {task_description}.

Your job: explore the wiki knowledge base and the execution traces, diagnose root causes of failures, and propose
ONE skill change (create a skill, patch an existing skill, retire a skill, or no_action).

## Tools
{tool_protocol}
- read_index {{}}: wiki/index.md, the catalogue of known patterns. Start here.
- read_impact {{}}: wiki/skill-impact.md, everything tried before with outcomes. Contains the full content of recently
  rejected proposals. DO NOT repeat a rejected approach.
- read_log {{}}: the chronological evolution log.
- read_pattern {{"name": "<pattern-name>"}}: one pattern page with root cause, evidence and workarounds.
- list_traces {{"status": "failing" | "passing" | "all"}}: every training task with PASS/FAIL, prediction and truth.
- read_trace {{"id": "<trace id>", "include_media": false}}: the full execution trace. Set include_media true only when
  the images themselves are needed to diagnose the failure (they are costly).
- read_skill {{"name": "<skill>"}}: the current SKILL.md and PURPOSE.md of an active skill.
- finish {{"proposal": {{...}}}}: submit your proposal (format below).

## Workflow
1. read_index, then read_impact.
2. Read the pattern pages relevant to the current failures.
3. Read execution traces of failed tasks (at least {min_trace_reads}) to confirm root causes; compare with a passing one.
4. Decide: create, patch, retire, or no_action. Prefer patching an existing skill when it is partially right.
5. finish.

## Skill layers
Skills are grouped in layers, outer to inner (for example a shared catalogue, then a team, then a project). You may
create, patch and retire skills only in EDITABLE layers; new skills land in the innermost editable layer
({editable_layer}). A skill in a READ-ONLY layer cannot be edited: if it is wrong for this workspace, use action
"override" to add an inner skill that replaces it here, citing the outer skill in "overrides". Skills already
overridden are hidden from the agent; change the overriding skill instead.{promotion_note}

## Proposal format
Create a new skill:
{{"action": "create", "name": "snake_case_name", "skill_md": "---\\nname: snake_case_name\\ndescription: one line\\n---\\n# Title\\n## When to apply\\n...\\n## When NOT to apply\\n...\\n## Instructions\\n...", "purpose_md": "## Origin\\n...\\n## Patterns addressed\\n...\\n## Evolution history\\n...", "rationale": "...", "citations": ["pattern-name", "trace-id"]}}
Merge skills into one: same as create, plus "supersedes": ["old_skill_a", "old_skill_b"].
Patch an existing skill (keep edits small and targeted):
{{"action": "patch", "name": "existing_skill", "edits": [{{"op": "append" | "replace" | "insert_after", "target": "exact text", "content": "..."}}], "rationale": "...", "citations": [...]}}
Retire a skill: {{"action": "retire", "name": "existing_skill", "rationale": "..."}}
Override a read-only skill: same fields as create, plus "overrides": "outer_skill_name".{promotion_format}
Nothing worth changing: {{"action": "no_action", "rationale": "..."}}

## Rules
1. Read the wiki FIRST and never re-propose something skill-impact shows was rejected.
2. Focus on concrete action patterns and strategies the executing agent can follow.
3. Keep skills concise and actionable; state when to apply and when NOT to apply.
4. Skills marked protected cannot be changed; work around them.
5. One tool call per turn; you have at most {max_turns} turns."""


class SkillProposer:
    def __init__(self, model: ChatModel, config: EvolveConfig):
        self.model, self.config = model, config

    def _opening(self, wiki: WikiState, skills: SkillSet, traces: list[Trace], feedback: dict[str, Any], iteration: int) -> str:
        overridden = skills.overridden()
        skill_lines = []
        for name, s in sorted(skills.skills.items()):
            notes = []
            if skills.multi_layer:
                notes.append(f"layer {s.layer}" + ("" if skills.layer(s.layer).editable else ", read-only"))
            if s.protected:
                notes.append("protected")
            if name in overridden:
                notes.append(f"overridden by {overridden[name]}")
            if s.overrides:
                notes.append(f"overrides {s.overrides}")
            suffix = f" ({'; '.join(notes)})" if notes else ""
            skill_lines.append(f"- {name}{suffix}: {s.description or '(no description)'}")
        skill_lines = skill_lines or ["(no active skills yet)"]
        return "\n\n".join(
            [
                f"# Iteration {iteration}",
                "## Active skills\n" + "\n".join(skill_lines),
                "## Training outcomes summary\n" + summarize_outcomes(traces),
                *(["## Skill usage across the training traces (which skills the agent had in context, by task outcome)\n" + usage]
                  if (usage := summarize_skill_usage(traces)) else []),
                "## Aggregate feedback from the previous validation\n"
                + (json.dumps(feedback, ensure_ascii=False, sort_keys=True, default=str) if feedback else "(none)"),
                "## Wiki index (from read_index)\n" + wiki.render_index(),
                "Begin. Reply with one tool call.",
            ]
        )

    @staticmethod
    def _call_from(response) -> ToolCall:
        """One tool call per turn, from a native invocation when present, else from the text (same semantics)."""
        if response.tool_calls:
            if len(response.tool_calls) != 1:
                raise ToolCallError(f"call exactly one tool per turn (got {len(response.tool_calls)})")
            invocation = response.tool_calls[0]
            if invocation.name not in TOOLS:
                raise ToolCallError(f"unknown tool {invocation.name!r}; allowed: {sorted(TOOLS)}")
            args = invocation.args if isinstance(invocation.args, dict) else {}
            return ToolCall(invocation.name, dict(args))
        return parse_tool_call(response.text, allowed=TOOLS)

    async def propose(
        self,
        wiki: WikiState,
        skills: SkillSet,
        traces: list[Trace],
        *,
        iteration: int,
        feedback: dict[str, Any] | None = None,
        rejected_candidates: dict[str, int] | None = None,
    ) -> tuple[Proposal, Scratchpad]:
        """``rejected_candidates`` maps a candidate content digest to the iteration it was rejected at; a proposal that
        reproduces one is refused inside the ReAct loop so the model revises instead of burning an iteration."""
        rejected_candidates = rejected_candidates or {}
        try:
            editable_layer = skills.innermost_editable.name
        except ProposalError:
            editable_layer = "none: every layer is read-only"
        native = bool(getattr(self.model, "supports_tools", False))
        system = SYSTEM_PROMPT.format(
            tool_protocol=NATIVE_PROTOCOL if native else TEXT_PROTOCOL,
            task_description=self.config.task_description,
            min_trace_reads=self.config.react_min_trace_reads,
            max_turns=self.config.react_max_turns,
            editable_layer=editable_layer,
            promotion_note=(
                " You may also \"promote\" an editable skill one layer outward when the evidence shows it applies beyond this workspace."
                if self.config.allow_promotions else ""
            ),
            promotion_format=(
                '\nPromote a skill one layer outward: {"action": "promote", "name": "existing_skill", "rationale": "..."}'
                if self.config.allow_promotions else ""
            ),
        )
        opening = self._opening(wiki, skills, traces, feedback or {}, iteration)
        by_id = {t.id: t for t in traces}
        pad = Scratchpad()
        trace_reads: set[str] = set()
        images_left = self.config.images_per_run
        required_reads = min(self.config.react_min_trace_reads, len(traces))

        for _turn in range(self.config.react_max_turns):
            parts: list[Part] = [TextPart(opening)]
            if pad.turns:
                parts = [TextPart(opening), TextPart("\n\n## Transcript so far\n"), *pad.parts(char_cap=self.config.scratchpad_char_cap)]
            response = await self.model.complete(
                ModelRequest(role="skill_proposer", system=system, parts=parts, max_tokens=self.config.max_tokens, tools=TOOL_SPECS if native else ())
            )
            try:
                call = self._call_from(response)
            except ToolCallError as exc:
                pad.add(None, response.text, f"ERROR: {exc}")
                continue
            shown = response.text or call.render()  # a native call has no text; the transcript still shows what was called
            observation, media = "", []
            if call.name == "read_index":
                observation = wiki.render_index()
            elif call.name == "read_impact":
                observation = wiki.render_impact(full_bodies=self.config.impact_full_bodies)
            elif call.name == "read_log":
                observation = "\n".join(wiki.render_log().splitlines()[-120:])
            elif call.name == "read_pattern":
                pid = normalise_pattern_id(str(call.args.get("name", "")))
                pattern = wiki.patterns.get(pid)
                observation = pattern.to_markdown() if pattern else f"ERROR: unknown pattern {pid!r}. Known: {sorted(wiki.patterns)}"
            elif call.name == "list_traces":
                status = str(call.args.get("status", "all"))
                subset = [
                    t for t in traces if status == "all" or (status == "failing") == (not t.outcome.passed)
                ]
                observation = summarize_outcomes(subset)
            elif call.name == "read_trace":
                trace = by_id.get(str(call.args.get("id", "")))
                if trace is None:
                    observation = f"ERROR: unknown trace id {call.args.get('id')!r}; use list_traces to see ids"
                else:
                    trace_reads.add(trace.id)
                    observation = trace.rendered(char_cap=self.config.trace_char_cap)
                    if call.args.get("include_media") and trace.media:
                        take = min(self.config.images_per_read, images_left, len(trace.media))
                        media = [
                            ImagePart(m.media_type, m.data_base64, ref=m.ref or f"{trace.id}:image{j + 1}")
                            for j, m in enumerate(trace.media[:take])
                        ]
                        images_left -= take
                        if take < len(trace.media):
                            observation += f"\n[{len(trace.media) - take} further image(s) not attached: media budget]"
            elif call.name == "read_skill":
                skill = skills.skills.get(str(call.args.get("name", "")))
                if skill is None:
                    observation = f"ERROR: unknown skill {call.args.get('name')!r}. Active: {sorted(skills.skills)}"
                else:
                    status = [f"layer: {skill.layer}", "editable" if skills.layer(skill.layer).editable else "read-only (override to change)"]
                    if skill.protected:
                        status.append("protected")
                    if skill.overrides:
                        status.append(f"overrides {skill.overrides}")
                    if skill.name in skills.overridden():
                        status.append(f"overridden by {skills.overridden()[skill.name]} (hidden from the agent)")
                    in_play = [t for t in traces if skill.name in t.skills_in_play]
                    if in_play:
                        failing = sum(not t.outcome.passed for t in in_play)
                        status.append(f"in play in {len(in_play)} training trace(s): {failing} failing, {len(in_play) - failing} passing")
                    elif any(t.skills_in_play for t in traces):
                        status.append("not in play in any training trace")
                    observation = f"[{'; '.join(status)}]\n<<< SKILL.md >>>\n{skill.render_skill_md()}\n<<< PURPOSE.md >>>\n{skill.purpose}"
            elif call.name == "finish":
                raw = call.args.get("proposal", call.args)
                try:
                    proposal = Proposal.from_dict(raw, allow_promotions=self.config.allow_promotions)
                    if proposal.action != "no_action" and len(trace_reads) < required_reads:
                        raise ProposalError(
                            f"you have read {len(trace_reads)} trace(s); read at least {required_reads} with read_trace before proposing a change"
                        )
                    candidate = apply_proposal(skills, proposal, iteration=iteration, allow_protected=self.config.allow_protected_edits)
                    if proposal.action not in ("no_action", "promote"):  # a promote moves a skill between layers; the text is unchanged by design
                        digest = candidate.content_digest()
                        if digest == skills.content_digest():
                            raise ProposalError("this proposal changes nothing the agent reads; propose a real change or no_action")
                        if digest in rejected_candidates:
                            raise ProposalError(
                                f"this candidate is identical to the one REJECTED at iteration {rejected_candidates[digest]} "
                                "(see read_impact); revise it or choose no_action"
                            )
                except ProposalError as exc:
                    observation = f"ERROR: proposal rejected: {exc}. Fix it and call finish again, or choose no_action."
                else:
                    pad.add(call, shown, "proposal accepted for validation")
                    return proposal, pad
            pad.add(call, shown, observation, media)

        return Proposal(action="no_action", rationale="ReAct turn budget exhausted without a valid finish() call"), pad


__all__ = ["NATIVE_PROTOCOL", "SYSTEM_PROMPT", "TEXT_PROTOCOL", "TOOLS", "TOOL_SPECS", "SkillProposer"]
