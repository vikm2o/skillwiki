# Changelog

## 0.2.0 (2026-09-13)

Production hardening. Every departure from the paper is catalogued in `docs/paper-differences.md`.

- Review fixes (2026-09-13, second pass): model-authored `SKILL.md` frontmatter can no longer crash the run. A YAML
  list or nested block, and a key the renderer cannot write back (`When to use:`), are `ProposalError`s the proposer
  sees inside its ReAct loop instead of a bare `ValueError` escaping `evolve` at `unified_diff`. `frontmatter.split`
  accepts CRLF. `SkillSet.content_digest()` covers `overrides` (set only), so an override is not refused as a duplicate
  of a plain create with the same body. `SkillSet.render_for_prompt(effective=True)` is the agent's view (overridden
  skills omitted); the README example uses it. A proposal the harness cannot apply counts toward the rejection streak
  (`reason: proposal_not_applied`), as `no_action` does.

- Documents carry `"schema": 2`; 0.1 documents load unchanged.
- Skill layers: ordered, outer to inner, each editable or read-only. New proposal actions `override` (shadow a
  read-only skill from an inner layer) and `promote` (opt-in via `EvolveConfig.allow_promotions`). Single-layer sets
  render exactly as before.
- `PairedGate`: paired-bootstrap acceptance over per-task outcomes. The paper's strict gate stays the default.
- `Budget` on `EvolveConfig`: model calls, output tokens, evaluations, wall-clock. Enforced before each paid call;
  exhaustion stops the run cleanly with `stopped_reason="budget_exhausted"` and usage on the report.
- The host's materialised candidate (from `SkillStore.propose`) is what gets validated and kept; a content
  mismatch with the proposal is a warning.
- Impact entries record pattern `citations`, the candidate's `content_digest` and the layer the change landed in.
- `WikiPruner` role (off by default): proposes merges of same-root-cause patterns and retirements of dead ones,
  applied under hard rules (protected patterns, shown bodies only); retired patterns are kept in `wiki/retired.md`; the maintainer is told not to recreate them.
- Resume: `CheckpointStore` (`RevisionCheckpointStore`, `NullCheckpointStore`); an interrupted iteration is finished
  on restart without repeating the maintainer and proposer; a recorded validation is reused.
- Known candidates are refused before validation: inside the proposer's `finish` (revise in the same loop) and as a
  harness backstop, keyed on the candidate content digest recorded in the impact log.
- `Trace.skills_in_play` and `summarize_skill_usage()`: the roles see which skills were in context per task and a
  per-skill failing/passing usage table; `read_skill` reports the skill's usage.
- `evolve(..., redact=)` with `regex_redactor()` / `EMAIL`: trace text is redacted before the roles and the trace store.
- `skillwiki` command line (`show`, `export`, `impact`, `history`, `diff`) over `file:` and `postgres:` stores.
- Native tool calling: `ToolSpec`/`ToolInvocation` on the model protocol; a `ChatModel` with `supports_tools` gets the
  proposer's tools as schemas and answers natively, with identical semantics to the text protocol. Anthropic and
  OpenAI adapters implement it; `ScriptedChatModel(native_tools=True)` for tests.
- Wiki transfer: `WikiState.inherit_from`, `seed_workspace()`, CLI `transfer --to`. Inherited citations carry
  `meta.source_workspace`; impact history is never inherited.
- Review fixes before the first tag: `promote` is accepted structurally (validation cannot distinguish it from the
  incumbent); `PairedGate` no longer treats two `ref=None` evaluations as the probe; output tokens are counted under
  the OpenAI name too; a merge into a pattern retired in the same prune update is refused; `SKILL.md` round-trips
  `protected`, `layer` and `overrides` (honoured only with `parse_skill_md(trusted=True)`: model-authored frontmatter
  can never protect, promote or override a skill); a zero budget stops cleanly and its report serialises; the blob
  store tolerates a retry of its own write.

## 0.1.0 (2026-09-12)

First release. Independent reimplementation of WikiSkill (arXiv:2608.27454) as an importable package.

- Three-layer workspace as immutable, digest-chained JSON documents; the paper's `wiki/` and `skills/` file layout
  is a rendering (`to_markdown_files`, `export_workspace`).
- Wiki Maintainer (incremental patch edits, index, log) and ReAct Skill Proposer (structured-text tool protocol,
  one atomic proposal per iteration) following the paper's prompts in structure.
- Algorithm 1 harness with pluggable `TraceSource`, `Evaluator`, `Gate` and `Hooks`; stratified 5/3 sampling with
  rotation; skill-impact tracker with full bodies of recent rejections.
- Extensions: onboarded baseline skills with `protected`, host-invalidated evidence marked `stale_evidence`,
  multimodal traces with per-turn media attachment.
- Stores: memory, file, PostgreSQL (`[postgres]`), S3 (`[s3]`), GCS (`[gcs]`), all fork-free.
- Optional `ChatModel` adapters for Anthropic and OpenAI-compatible APIs.
