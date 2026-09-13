# skillwiki

Compile agent experience into a persistent wiki that drives gated skill evolution.

`skillwiki` is an independent reimplementation of the loop described in
*WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution*
(Tang, Rashtchian, Ferng, Tomkins, Juan, Vu; Google Research; [arXiv:2608.27454](https://arxiv.org/abs/2608.27454)).
It is not the authors' code and contains no third-party code. Apache-2.0.
Written by [Vikash Ranjan](https://www.linkedin.com/in/vikash-ranjan-stylsai/), CTO, [styls.ai](https://styls.ai).

## What the paper does, and what this package keeps

The workspace has three layers. **Raw**: immutable execution traces. **Wiki**: a
persistent knowledge base of patterns (root causes and workarounds), an index, an
evolution log and a skill-impact tracker, never reset and never rolled back.
**Skills**: the active procedural instructions the agent reads at inference time.

Each iteration, an agent runs tasks with the current skills, a **Wiki Maintainer**
consolidates a stratified sample of failing and passing traces into the wiki, a
**Skill Proposer** explores the wiki and traces ReAct-style and proposes one atomic
skill change, and a **gate** accepts the change only if validation improves.
Rejected skills are rolled back; the wiki keeps what was learned either way.

`skillwiki` keeps that mechanism exactly and changes only what a production
deployment needs:

| Paper | skillwiki |
| --- | --- |
| Files on disk (`raw/`, `wiki/`, `skills/`) | Immutable JSON documents chained by SHA-256 digest behind store protocols. The paper's file layout is a rendering (`WikiState.to_markdown_files()`, `export_workspace()`), never the storage. |
| One filesystem | `memory`, `file` (local), `postgres` (recommended), `s3`, `gcs` adapters; the object-store adapters use backend preconditions so two writers cannot fork a chain. |
| Fresh rollouts every iteration | Any `TraceSource`: fresh rollouts, or production runs that later received a label, at no extra cost. |
| Text traces | Multimodal traces. Images travel only in the ReAct turn that fetched them; the folded transcript keeps a placeholder. |
| Native tool calling | Both protocols, same tools and rules: by default the model answers with one `{"tool": ..., "args": ...}` JSON object per turn and the harness sends the folded scratchpad as one user message (works on any provider, including metered transports that forbid tools); a model that sets `supports_tools` gets the same eight tools as native `ToolSpec`s instead. |
| Empty S₀ | Optional onboarded baseline skills, with a per-skill `protected` flag. |
| Validation score `>` best | Pluggable `Gate`. Default is the paper's strict improvement; `PairedGate` (recommended in production) accepts only when a paired bootstrap over per-task outcomes says the gain is not noise. |
| One flat `skills/` directory | Ordered skill **layers** (e.g. global → tenant → project), each editable or read-only. Read-only skills are `override`d, never edited; `render_for_prompt(effective=True)` is what the agent reads (overridden skills omitted); single-layer sets render exactly as the paper's layout. |
| Bounded by iterations and rejections | Also bounded by a `Budget`: model calls, output tokens, evaluations, wall-clock. Exhaustion stops the run cleanly and keeps the wiki work done so far. |
| The wiki only grows | Optional `WikiPruner` merges same-cause patterns and retires evidence-less ones under hard rules; nothing is deleted (`wiki/retired.md`). |
| One process per run | `CheckpointStore` resume: an interrupted iteration is finished, not repeated. |
| Every proposal is validated | A candidate identical to one already rejected (or to the current head) is refused before validation. |
| Evidence never changes | Hosts can invalidate evidence (a corrected label); affected citations are marked `stale_evidence`, never deleted. |
| The agent loads every skill | `Trace.skills_in_play` records what the agent actually had in context; the roles see per-skill usage on failing vs passing tasks. |
| Benchmark traces | `evolve(..., redact=)` strips identifiers from production traces before any role or store sees them. |
| Open the files | `skillwiki --store file:DIR show / export / impact / history / diff` reads a chained store the way the paper reads a directory. |
| One wiki | `WikiState.inherit_from` / `seed_workspace` / `skillwiki transfer`: a new workspace starts from another's patterns, with citations marked as foreign provenance and no inherited impact history. |

## Install

```bash
pip install "skillwiki @ git+https://github.com/vikm2o/skillwiki@v0.2.0"
pip install "skillwiki[postgres] @ git+https://github.com/vikm2o/skillwiki@v0.2.0"   # production store
```

Extras: `postgres`, `s3`, `gcs`, `anthropic`, `openai`, `dev`. The core has no dependencies.

## Sixty-second example

```python
import asyncio
from skillwiki import (EvolveConfig, Evaluation, StaticTraceSource, TaskOutcome, Trace, evolve)
from skillwiki.adapters.anthropic import AnthropicChatModel      # or implement ChatModel yourself
from skillwiki.stores import MemoryRevisionStore, RevisionSkillStore, RevisionWikiStore

class MyEvaluator:                                           # run the validation split with a skill set
    async def evaluate(self, ref, skill_set, *, iteration, purpose):
        score = run_my_agent_on_validation(skill_set.render_for_prompt(effective=True))  # the agent never sees overridden skills
        return Evaluation(ref=ref, score=score)

traces = [Trace(id="t1", outcome=TaskOutcome("task-1", 0.0, False, prediction="B", truth="C"), text="...full trajectory...")]
revisions = MemoryRevisionStore()
report = asyncio.run(evolve(
    config=EvolveConfig(task_description="multiple-choice maths questions"),
    model=AnthropicChatModel(model="claude-sonnet-5"),
    wiki_store=RevisionWikiStore(revisions, "maths"),
    skill_store=RevisionSkillStore(revisions, "maths"),
    trace_source=StaticTraceSource(traces),
    evaluator=MyEvaluator(),
))
print(report.to_dict())
print(report.wiki.render_index())
```

`examples/filesystem_toy.py` runs the whole loop with a scripted model and no API key and writes the paper's
`wiki/` and `skills/` directories so you can watch the wiki compound; `examples/anthropic_toy.py` does the same with a real model.

## Integrating with your own system

Implement four small protocols (all `async`):

- `TraceSource.collect(skill_set, *, iteration) -> list[Trace]` — where traces come from. A `Trace` carries the text
  rendering, optional images, a `TaskOutcome` (score, passed, prediction, truth) and opaque `meta` that is copied onto
  every wiki citation of it, so your provenance (label ids, asset ids) rides along.
- `Evaluator.evaluate(ref, skill_set, *, iteration, purpose) -> Evaluation` — run the validation split.
- `Gate.decide(best, candidate) -> Decision` — accept or reject. Return `stop=True` to end early.
- `ChatModel.complete(ModelRequest) -> ModelResponse` — one system prompt, one user message of text and image parts, text back.

Stores: implement `RevisionStore` (four methods: `head`, `append`, `get`, `list`) and use `RevisionWikiStore` / `RevisionSkillStore` /
`RevisionTraceStore`, or implement `WikiStore` / `SkillStore` / `TraceStore` directly against your own tables.
`SkillStore.propose()` is where a host materialises a candidate in its own format (a bundle, a package, a row) and
`Hooks.stage()` is where it wraps paid stages in leases, idempotent checkpoints and spend metering.

Skills are markdown (`SKILL.md` frontmatter + body, `PURPOSE.md`). Hosts with structured skills write a small codec
that maps fields to frontmatter; the module only ever patches the body.

## Fidelity notes

- Sampling: up to 5 failing and 3 passing traces per iteration, 15,000 characters each (paper App. C). When the
  trace set does not change between iterations, sampling rotates without replacement so iterations see new evidence.
- Proposer: is told to read the index, then the skill-impact tracker, then traces; the harness enforces at least four trace reads before a change may be proposed (App. E.3).
  One atomic proposal per iteration; a merge is a `create` with `supersedes`.
- Skill-impact: full content of the last five rejected proposals, diffs and one-line summaries beyond.
- Gate default: accept iff the validation score strictly improves; stop at 1.0 (Alg. 1). Use `PairedGate` when your
  evaluator returns per-task outcomes: it needs ≥20 shared tasks and a 90% bootstrap win probability by default.
- Stopping: 8 iterations or 3 consecutive rejections by default.

Every difference from the paper, with the reason for it, is listed in [`docs/paper-differences.md`](docs/paper-differences.md).

## Development

```bash
uv venv && uv pip install -e ".[dev,postgres,s3]"
ruff check src tests && pytest -q          # postgres and s3 tests need Docker / moto
```
