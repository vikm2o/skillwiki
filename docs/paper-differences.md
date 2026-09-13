# skillwiki versus the WikiSkill paper

`skillwiki` is an independent reimplementation of *WikiSkill: Compiling Agent Experience into Persistent Knowledge
for Skill Evolution* (Tang, Rashtchian, Ferng, Tomkins, Juan, Vu; arXiv:2608.27454). This document is the complete
list of where the package follows the paper exactly, where it departs, and why. It is the source for the release
write-up, so every departure states the production problem it solves.

Section references are to the paper (v1). "Alg. 1" is Appendix A; the prompts are Appendix E.

## 1. Kept exactly

| Mechanism | Paper | skillwiki |
| --- | --- | --- |
| Three layers | `raw/` immutable traces, `wiki/` persistent knowledge, `skills/` active instructions (§3.1) | `TraceStore`, `WikiStore`, `SkillStore`; same responsibilities, same immutability rules |
| Loop | Alg. 1: collect traces → maintain wiki → propose one skill change → validate → accept iff better | `harness.evolve`, one iteration per Alg. 1 loop body, same order |
| Agent isolation | The inference agent never reads the wiki during rollouts (§3.2.1, ablation §5.1) | `TraceSource.collect` and `Evaluator.evaluate` receive the skill set only; the wiki object is never passed to them |
| Wiki Maintainer | One call over a stratified sample; incremental edits; index rewrite; log append (§3.2.2, App. E.2) | `roles/maintainer.py`; prompt keeps the paper's sections (deep trace analysis, documentation rules, index quality) |
| Sampling | ≤5 failing + ≤3 passing traces, 15,000 characters each (App. C) | `EvolveConfig.failing_sample=5`, `passing_sample=3`, `trace_char_cap=15000` |
| Skill Proposer | ReAct agent; reads index first, then skill-impact, then ≥4 traces; one atomic proposal or no action (§3.2.3, App. E.3) | `roles/proposer.py`; the ≥4 reads are enforced by the harness, not just requested |
| Patch operations | `append`, `replace`, `insert_after` with exact unique targets (App. E) | `patch.py`, identical semantics |
| Skill-impact tracker | Every gated proposal with diff, validation result and outcome; full bodies of recent rejections (§3.2.4) | `ImpactEntry`; full bodies for the last 5 rejections, diffs beyond |
| Gate | Accept iff R_val strictly improves; early stop at a perfect score (Alg. 1 l.13) | `StrictImprovementGate`, the default |
| Rollback | Rejected skill sets are discarded; the wiki is never rolled back (§3.2.4) | Candidates live outside the skill chain until accepted; `WikiState` has no rollback path |
| Stopping | 8 iterations or 3 consecutive rejections (a `no_action` proposal also counts toward the streak, since it makes the same progress as a rejection: none) (App. C) | `max_iterations=8`, `max_rejected_streak=3` |
| File layout | `wiki/index.md`, `wiki/log.md`, `wiki/skill-impact.md`, `wiki/patterns/<id>.md`, `skills/<name>/SKILL.md` + `PURPOSE.md` | Produced by `to_markdown_files()` / `to_files()` / `export_workspace()` as a rendering of the stored documents |

## 2. Departures present since 0.1

Each entry: what the paper does, what skillwiki does, why.

### 2.1 Storage is immutable JSON, not files
Paper: three directories on one filesystem. skillwiki: JSON documents chained by SHA-256 digest behind a
`RevisionStore` protocol, with memory, file, PostgreSQL, S3 and GCS adapters. The file layout is a rendering.
Why: production runs on clusters with no shared disk; two workers must not fork a chain (`expected_head` /
`HeadMoved`, backed by unique `(workspace, kind, seq)` in Postgres and write preconditions on object stores).

### 2.2 Traces may come from production, not only fresh rollouts
Paper: every iteration re-runs the training tasks with the current skills. skillwiki: `TraceSource` is a protocol.
A host can return served production runs that later received a ground-truth label. Why: a production system already
pays for those runs; they are rollouts of the current skill set at zero extra cost, and they track the real input
distribution rather than a fixed training split.

### 2.3 Multimodal traces
Paper: text traces. skillwiki: traces carry images; the maintainer sees a bounded number, the proposer fetches them
per trace on request, and images are attached only in the turn that fetched them (a placeholder remains in the
folded transcript). Why: QC of generated images cannot be diagnosed from text alone; attaching every image on every
ReAct turn grows cost quadratically.

### 2.4 Structured-text tool protocol
Paper: native tool calling. skillwiki: the proposer emits one `{"tool", "args"}` JSON object per turn in plain text
and the harness re-sends the folded scratchpad as one user message. Why: metered or audited transports commonly allow
exactly one user message and no tool definitions. Native tool calling is an optional adapter (see §3.11).

### 2.5 Onboarded baseline skills and `protected`
Paper: S₀ = ∅. skillwiki: `initial_skills` seed the chain as impact iteration 0 with `origin: onboarded`; a skill
marked `protected` cannot be patched or retired. Why: real deployments start from human-written checklists that must
not be silently rewritten, and the baseline validation score should reflect them.

### 2.6 Pluggable gate
Paper: strict scalar improvement. skillwiki: `Gate` protocol; hosts plug in paired bootstraps, safety floors or cost
policies. Why: see §3.3.

### 2.7 Host-invalidated evidence
Paper: evidence is fixed. skillwiki: a host can report trace ids whose labels were later corrected; every citation of
them moves to `stale_evidence` and the maintainer is told to revise or narrow the claim. Nothing is deleted.
Why: production labels change; a wiki claim must not silently rest on retracted evidence, and deleting the pattern
would erase the reasoning trail.

### 2.8 Stratified sampling rotates
Paper: sample from a fresh rollout each iteration. skillwiki: when the trace set is unchanged between iterations,
sampling rotates through unseen traces first (seeded), resetting a stratum only when exhausted. Why: with production
traces the set often does not change between iterations, and re-sampling the same eight traces teaches nothing.

## 3. Departures added in 0.2

### 3.1 Documents carry a schema version
Paper: files, no versioning. skillwiki: every wiki, skill-set and trace document carries `"schema": 2`; loaders accept
unversioned 0.1 documents and upgrade them in memory. Why: a stored chain outlives the code that wrote it. Without a
version, the first incompatible change has to guess what an old document meant.

### 3.2 Skill layers, overrides and promotion
Paper: one flat `skills/` directory owned by one loop. skillwiki: a skill set is an ordered list of layers, outer to
inner (for example `global` → `tenant` → `project`), each editable or read-only. The proposer creates, patches and
retires only in editable layers. A read-only skill that is wrong for this workspace is *overridden* by an inner skill
that names it; the outer skill stays in the set, hidden from the agent (`SkillSet.render_for_prompt(effective=True)` is the
agent's rendering; the default is the roles' view, which shows the hidden skill with a note), visible in the diff and the
impact log. An
optional `promote` action moves a skill one layer outward when the host allows it. A single-layer set renders exactly
as the paper's layout, so the flat case is unchanged.
Why: production skills are shared. A per-workspace loop must be able to specialise a shared rule without silently
rewriting it for everyone, and the owner of the shared layer must be able to see where projects diverged. Before
this, a project run could shadow a global skill with no record that it had.

A `promote` is accepted structurally, without validation, when the host's skill store accepts it: it moves a skill
outward without changing a character the agent reads, so no validation score could separate it from the incumbent.

The file rendering carries `protected`, `layer` and `overrides` so an exported skill set round-trips, but the parser
honours those fields only when the caller marks the text as trusted (a host reading its own export). A skill body
authored by the proposer is never trusted: the layer and protection of a created skill come from the harness, so a
model cannot write `protected: true` or `layer: global` into a SKILL.md and have the loop lock or promote it.

### 3.3 A statistical gate
Paper: accept iff the validation mean strictly improves. skillwiki: `PairedGate` pairs the two evaluations by task,
bootstraps the paired differences and accepts only when the mean gain is positive and at least 90% of resamples
agree; below 20 shared tasks it refuses to decide. The paper's rule remains the default gate for fidelity.
Why: on a few hundred validation tasks the strict rule accepts noise roughly half the time, and every noisy accept
raises the bar the next candidate must beat, so the loop drifts on luck. The gate is where a wrong decision compounds.

### 3.4 Budgets
Paper: bounded by iteration count and rejection streak only. skillwiki: `Budget(max_model_calls, max_output_tokens,
max_evaluations, max_seconds)` checked before every model call and every validation. Exhaustion stops the run with
`stopped_reason="budget_exhausted"`, persists the wiki work already done, and records usage on the report.
Why: a ReAct proposer against a paid API is the first thing a new adopter runs; without a ceiling a mis-scripted
loop spends until the iteration cap. Hosts with their own metering keep it as a second fence.

### 3.5 The host's candidate is canonical
Paper: the proposer's output is the candidate. skillwiki: after `SkillStore.propose` materialises the candidate, the
harness validates and continues with the host's skill set, warning when the instructions or overrides differ from the
proposal (host-added identifiers and metadata are expected and not flagged).
Why: hosts normalise (re-key rows, validate schemas, collapse an override into a successor). Validating the module's
version while serving the host's would grade one artefact and ship another.

### 3.6 Wiki pruning
Paper: the wiki only grows. skillwiki: a `WikiPruner` role runs on a cadence (`prune_every`) or when the rendered
wiki exceeds `wiki_char_budget`. It sees the index plus one statistics line per pattern and, within a separate
budget, recent page bodies. It may merge patterns with the same root cause (evidence is unioned onto the survivor)
and may retire patterns it judges dead (all evidence stale, never cited, long untouched). The judgement is the
model's; the enforcement is these hard rules, which the model cannot override: a pattern cited by an accepted
skill is never retired or merged away; a body may be rewritten only if the pruner was shown it; retired patterns
move to `wiki/retired.md` with title, body, evidence and reason, and the maintainer is told not to recreate them.
Off by default so the paper's behaviour is the default.
Why: the maintainer reads the whole wiki every iteration. In a long-running deployment the context grows until the
model's window is exceeded and the loop stalls silently. Pruning is the only way to run the loop for months.

### 3.7 Resume

A resumed iteration calls `SkillStore.propose` again for the same (iteration, proposal), so a host that mints a
durable artefact there must make it idempotent on the iteration key rather than append blindly.
Paper: a run is one process. skillwiki: a `CheckpointStore` records, per iteration, the proposal and the module's
candidate once the wiki update is persisted, and the validation result once it is known. A restart on the same
stores finishes that iteration (re-materialising the candidate through the host, reusing a recorded validation)
instead of repeating the maintainer and proposer. The wiki was already saved before the proposal, so nothing the
maintainer learned is lost either way.
Why: a run spanning hours of production feedback collection will be interrupted by deploys and worker restarts.
Repeating an iteration costs two role calls and a validation; losing it costs the evidence the maintainer already
consolidated. Hosts without a re-entry path keep `NullCheckpointStore` and lose at most one proposal.

### 3.8 Known candidates are not re-validated
Paper: every proposal is validated. skillwiki: every impact entry carries the candidate's content digest (names,
bodies, frontmatter minus host identifiers; purpose history excluded). The proposer's `finish` refuses a candidate
identical to one already rejected, naming the iteration, or identical to the current head, so the model revises
within the same ReAct loop. The harness applies the same check as a backstop and records the refusal in the impact
log without spending a validation.
Why: after a rejection streak the proposer tends to re-propose its favourite candidate. Validation is the expensive
step (a full replay of the validation split), and paying it twice for one candidate buys nothing.

### 3.9 Skills in play are visible to the roles

**Paper.** The maintainer and proposer see the active skill set and the traces; nothing in a trace says which skills
the agent actually had in context for that task, because the paper's agent loads every skill every time.

**skillwiki.** Production agents retrieve a few skills per task. `Trace.skills_in_play` names them, the trace header
prints them, and `summarize_skill_usage()` turns a trace set into a per-skill table (in play on failing tasks vs
passing tasks) for skills that appear in at least one trace. The proposer sees the table across all training traces
and, on `read_skill`, how often that skill was in play and with what outcome; the maintainer sees the table for the
sampled traces, placed after the traces so it is the first thing a tight budget trims.

**Why.** A skill that was in context on most failing tasks and few passing ones is the obvious candidate to patch; a
skill that was never retrieved cannot be blamed for anything. Without this the proposer edits blind. Hosts whose
agents load every skill leave the field empty and nothing changes in the prompts.

### 3.10 Redaction before the roles

**Paper.** Traces are the benchmark's own rollouts; there is no one to protect.

**skillwiki.** `evolve(..., redact=)` applies a `Redactor` (`str -> str`) to every collected trace's text before
sampling, before the roles, and before the trace store records it. `regex_redactor({name: pattern})` builds one from
named patterns (`EMAIL` is provided); each match becomes `[REDACTED:name]` so the roles can still tell an e-mail from
an order number without seeing either. Structured outcome fields are left to the host, which owns their shape.

**Why.** Production traces carry reviewer notes, customer URIs and ticket ids. The model does not need them to find a
root cause, and the raw layer is a durable store.

### 3.11 A command line for the workspace

**Paper.** The artefacts are files: open `wiki/index.md` and read.

**skillwiki.** The artefacts are digest-chained documents, so the package ships `skillwiki --store file:DIR |
postgres:URL --workspace WS show | export DIR | impact | history | diff` to give an operator the same visibility. The
`postgres:` store needs the extra and fails with a plain message when it is missing.

**Why.** Immutable storage should not make the wiki harder to read than the paper's directory.

### 3.12 Native tool calling, same semantics

**Paper.** The proposer is a ReAct agent with native tool calls.

**skillwiki.** 0.1 carried tool calls inside the text so that any provider, and metered transports that forbid
`tools`, could run the proposer. 0.2 adds the native path without changing the loop: a `ChatModel` that sets
`supports_tools` receives the eight tools as JSON schemas (`ToolSpec`) and answers with `ToolInvocation`s; the
proposer turns either form into the same `ToolCall`, records it in the same scratchpad, and enforces the same rules
(one call per turn, known tools only, minimum trace reads). The Anthropic and OpenAI adapters do this; the scripted
test model can play either side, and one test runs the same ReAct script through both protocols and asserts the same
proposal. The maintainer and pruner never receive tools; they answer with one JSON object.

**Why.** Natively tool-calling models are more reliable at emitting well-formed calls than at embedding JSON in prose,
and the folded single-message transcript stays the contract, so a host can switch protocols without touching stores,
gates or prompts.

### 3.13 Wiki transfer between workspaces

**Paper.** One benchmark, one wiki.

**skillwiki.** `WikiState.inherit_from(other, *, source, evidence_meta=None)` copies the patterns (and retired
patterns) the target lacks, keeps their bodies and index lines, stamps every inherited citation
`meta.source_workspace`, sets `Pattern.inherited_from`, and writes one `[transfer]` log line. Impact history is not
copied: it records what the target validated. `seed_workspace()` and `skillwiki ... transfer --to` do this over a
revision store. The optional `evidence_meta` lets a host rewrite provenance on the way in: the first production host moves the source
scope's label revision ids to `source_label_revision_ids`, so a project's wiki depends on none of its tenant's labels.

**Why.** An organisation's wiki is the expensive part; a new project should start from it rather than rediscover it.
Inherited evidence belongs to the source workspace: the target's trace store cannot open it and its invalidation feed
cannot mark it stale, so it is labelled as foreign provenance rather than pretended to be local.

## 4. Deliberately unchanged

- The inference agent is never given the wiki. The paper's ablation shows it hurts; nothing here overrides it.
- One atomic proposal per iteration. Batching proposals would make the gate's attribution meaningless.
- The wiki is never rolled back, even when the skill it motivated is rejected.
- The module does not run the inference agent. Hosts own execution, cost and safety of their own agent.
