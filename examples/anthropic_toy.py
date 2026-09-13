# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""The same toy task as ``filesystem_toy.py`` with a real model (``pip install skillwiki[anthropic]``, ANTHROPIC_API_KEY set).

    python examples/anthropic_toy.py ./toy-workspace claude-sonnet-5

Costs a few dozen small calls. Watch how the maintainer names the pattern and whether the proposer's first skill
makes the toy agent subtract the discount: the evaluator only accepts skills whose body contains the word "subtract".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from skillwiki import EvolveConfig, evolve_sync
from skillwiki.adapters.anthropic import AnthropicChatModel
from skillwiki.stores import (
    FileRevisionStore,
    RevisionSkillStore,
    RevisionTraceStore,
    RevisionWikiStore,
    export_workspace,
)

sys.path.insert(0, str(Path(__file__).parent))
from filesystem_toy import ToyEvaluator, ToyTraces  # noqa: E402


def main(root: str, model: str) -> None:
    revisions = FileRevisionStore(Path(root) / "revisions")
    report = evolve_sync(
        config=EvolveConfig(
            task_description=("arithmetic word problems. The executing agent is a tiny rule-based program: it multiplies the two "
                              "numbers for plain problems and, for discount problems, only subtracts the discount when an active skill's "
                              "text contains the word 'subtract'."),
            max_iterations=4,
        ),
        model=AnthropicChatModel(model=model),
        wiki_store=RevisionWikiStore(revisions, "toy"),
        skill_store=RevisionSkillStore(revisions, "toy"),
        trace_store=RevisionTraceStore(revisions, "toy"),
        trace_source=ToyTraces(),
        evaluator=ToyEvaluator(),
    )
    export_workspace(root, wiki=report.wiki, skills=report.skills)
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "./toy-workspace", sys.argv[2] if len(sys.argv) > 2 else "claude-sonnet-5")
