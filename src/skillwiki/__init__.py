# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""skillwiki: compile agent experience into a persistent wiki that drives gated skill evolution.

Independent reimplementation of WikiSkill (Tang et al., arXiv:2608.27454) that is
storage-agnostic, provider-agnostic and multimodal. See README.md.
"""

from .config import Budget, EvolveConfig
from .documents import HeadMoved, Revision, canonical_json, digest
from .gates import Decision, Evaluation, Evaluator, Gate, PairedGate, StrictImprovementGate
from .harness import RunReport, evolve, evolve_sync
from .hooks import BudgetExceeded, Hooks, IterationReport
from .model import (
    ChatModel,
    ImagePart,
    ModelRequest,
    ModelResponse,
    ScriptedChatModel,
    TextPart,
    ToolInvocation,
    ToolSpec,
)
from .patch import PatchError, PatchOp, apply_patch
from .redaction import EMAIL, Redactor, redact_trace, regex_redactor
from .skills import Layer, Proposal, ProposalError, SkillDocument, SkillSet, apply_proposal, unified_diff
from .stores import (
    Candidate,
    CheckpointStore,
    MemoryRevisionStore,
    NullCheckpointStore,
    NullTraceStore,
    RevisionCheckpointStore,
    RevisionSkillStore,
    RevisionStore,
    RevisionTraceStore,
    RevisionWikiStore,
    SkillStore,
    TraceStore,
    WikiStore,
    seed_workspace,
)
from .traces import (
    StaticTraceSource,
    TaskOutcome,
    Trace,
    TraceSource,
    stratified_sample,
    summarize_outcomes,
    summarize_skill_usage,
)
from .wiki import (
    EvidenceRef,
    ImpactEntry,
    IndexEntry,
    LogEntry,
    MaintainerUpdate,
    Pattern,
    PruneUpdate,
    RetiredPattern,
    WikiError,
    WikiState,
)

__version__ = "0.2.0"
__author__ = "Vikash Ranjan, CTO, styls.ai"

__all__ = [
    "ToolInvocation",
    "ToolSpec",
    "seed_workspace",
    "EMAIL",
    "Redactor",
    "redact_trace",
    "regex_redactor",
    "summarize_skill_usage",
    "RetiredPattern",
    "PruneUpdate",
    "RevisionCheckpointStore",
    "NullCheckpointStore",
    "CheckpointStore",
    "PairedGate",
    "Layer",
    "BudgetExceeded",
    "Budget",
    "Candidate",
    "ChatModel",
    "Decision",
    "Evaluation",
    "Evaluator",
    "EvidenceRef",
    "EvolveConfig",
    "Gate",
    "HeadMoved",
    "Hooks",
    "ImagePart",
    "ImpactEntry",
    "IndexEntry",
    "IterationReport",
    "LogEntry",
    "MaintainerUpdate",
    "MemoryRevisionStore",
    "ModelRequest",
    "ModelResponse",
    "NullTraceStore",
    "PatchError",
    "PatchOp",
    "Pattern",
    "Proposal",
    "ProposalError",
    "Revision",
    "RevisionSkillStore",
    "RevisionStore",
    "RevisionTraceStore",
    "RevisionWikiStore",
    "RunReport",
    "ScriptedChatModel",
    "SkillDocument",
    "SkillSet",
    "SkillStore",
    "StaticTraceSource",
    "StrictImprovementGate",
    "TaskOutcome",
    "TextPart",
    "Trace",
    "TraceSource",
    "TraceStore",
    "WikiError",
    "WikiState",
    "WikiStore",
    "__author__",
    "__version__",
    "apply_patch",
    "apply_proposal",
    "canonical_json",
    "digest",
    "evolve",
    "evolve_sync",
    "stratified_sample",
    "summarize_outcomes",
    "unified_diff",
]
