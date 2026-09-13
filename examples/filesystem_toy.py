# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
"""The loop end to end with a scripted model and the file store. No API key, no network.

    python examples/filesystem_toy.py ./toy-workspace

The toy task: an "agent" answers arithmetic word problems and fails whenever a problem mentions a discount, because it
forgets to subtract. The scripted maintainer notices, the scripted proposer writes a skill, the evaluator (which
really re-scores the validation problems with the skill applied) accepts it, and the wiki records every step. Open
``<workspace>/wiki/`` and ``<workspace>/skills/`` afterwards: that is the paper's layout, rendered from the stored chain.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from skillwiki import Evaluation, EvolveConfig, PairedGate, ScriptedChatModel, TaskOutcome, Trace, evolve_sync
from skillwiki.stores import (
    FileRevisionStore,
    RevisionSkillStore,
    RevisionTraceStore,
    RevisionWikiStore,
    export_workspace,
)

PROBLEMS = [
    ("p1", "A shirt costs 40 and has a 10 discount. Total?", 30, "discount"),
    ("p2", "3 pens at 4 each. Total?", 12, "plain"),
    ("p3", "A bag costs 25 with a 5 discount. Total?", 20, "discount"),
    ("p4", "2 books at 9 each. Total?", 18, "plain"),
    ("p5", "Shoes cost 60 with a 15 discount. Total?", 45, "discount"),
    ("p6", "5 apples at 2 each. Total?", 10, "plain"),
]
VALIDATION = [("v1", "A hat costs 20 with a 5 discount. Total?", 15, "discount"), ("v2", "4 mugs at 3 each. Total?", 12, "plain"),
              ("v3", "A coat costs 90 with a 30 discount. Total?", 60, "discount"),
              ("v4", "A scarf costs 10 plus 2 tax. Total?", 12, "tax")]  # nobody teaches tax, so validation never reaches 1.0


def toy_agent(problem: str, kind: str, skill_text: str) -> int:
    """Forgets the discount unless a skill mentions subtracting it."""
    numbers = [int(tok) for tok in problem.replace(".", " ").replace("?", " ").split() if tok.isdigit()]
    if kind == "discount":
        return numbers[0] - numbers[1] if "subtract" in skill_text.lower() else numbers[0]
    if kind == "tax":
        return numbers[0]
    return numbers[0] * numbers[1]


class ToyTraces:
    async def collect(self, skill_set, *, iteration):
        skill_text = skill_set.render_for_prompt()
        traces = []
        for pid, problem, truth, kind in PROBLEMS:
            answer = toy_agent(problem, kind, skill_text)
            traces.append(Trace(id=f"{pid}-it{iteration}", outcome=TaskOutcome(pid, float(answer == truth), answer == truth, prediction=answer, truth=truth),
                                text=f"Problem: {problem}\nAgent reasoning: read numbers {problem}\nAgent answer: {answer}\nSkills seen:\n{skill_text}"))
        return traces


class ToyEvaluator:
    async def evaluate(self, ref, skill_set, *, iteration, purpose):
        skill_text = skill_set.render_for_prompt()
        outcomes = [TaskOutcome(pid, float(toy_agent(problem, kind, skill_text) == truth), toy_agent(problem, kind, skill_text) == truth)
                    for pid, problem, truth, kind in VALIDATION]
        return Evaluation(ref=ref, score=sum(o.score for o in outcomes) / len(outcomes), per_task=outcomes)


def tool(name, **args):
    return json.dumps({"tool": name, "args": args})


def script(request):
    text = request.text
    if request.role == "wiki_maintainer":
        iteration = int(text.split("# Iteration ")[1].split("\n")[0])
        failing = [line.split()[2] for line in text.splitlines() if line.startswith("=== Trace ") and "FAIL" in line]
        if iteration == 1:
            return json.dumps({
                "create_patterns": [{"name": "forgotten-discount", "title": "Discount not subtracted", "evidence": failing[:3],
                                     "content": "# Discount not subtracted\nRoot cause: the agent reads the price and stops; it never subtracts the discount.\n"
                                                "Failing: 'costs 40 ... 10 discount' -> 40 (truth 30).\nFix: when the problem says discount, subtract it from the price."}],
                "update_patterns": [], "append_log": "Iteration 1: all failures are discount problems.",
                "update_index": "- [forgotten-discount](wiki/patterns/forgotten-discount.md): discount problems fail; agent never subtracts; subtract the discount.",
            })
        return json.dumps({"create_patterns": [], "update_patterns": [{"name": "forgotten-discount", "evidence": failing[:1] or [],
                           "edits": [{"op": "append", "content": f"Iteration {iteration}: {'still failing' if failing else 'no failures observed'}."}]}] if True else [],
                           "append_log": f"Iteration {iteration}: {len(failing)} failing traces.",
                           "update_index": "- [forgotten-discount](wiki/patterns/forgotten-discount.md): discount problems fail; agent never subtracts; subtract the discount."})
    turns = text.count("### Turn ")
    if turns == 0:
        return tool("read_index")
    if turns == 1:
        return tool("read_impact")
    ids = [line.split()[1] for line in text.splitlines() if line.startswith("- ") and ("[PASS]" in line or "[FAIL]" in line)]
    if turns - 2 < min(4, len(ids)):
        return tool("read_trace", id=ids[turns - 2])
    if "ACCEPTED" in text and "apply_discounts" in text:
        return tool("finish", proposal={"action": "no_action", "rationale": "the discount skill is accepted; nothing new in the wiki"})
    return tool("finish", proposal={
        "action": "create", "name": "apply_discounts", "rationale": "forgotten-discount pattern",
        "citations": ["forgotten-discount"],
        "skill_md": "---\nname: apply_discounts\ndescription: Subtract any discount from the price before answering\n---\n"
                    "# Apply discounts\n## When to apply\nThe problem mentions a discount.\n## Instructions\nSubtract the discount from the price. Total = price - discount.\n",
        "purpose_md": "## Origin\nPattern forgotten-discount (iteration 1).\n## Patterns addressed\nforgotten-discount\n## Evolution history\n",
    })


def main(root: str) -> None:
    revisions = FileRevisionStore(Path(root) / "revisions")
    report = evolve_sync(
        config=EvolveConfig(task_description="arithmetic word problems", max_iterations=3, react_min_trace_reads=4),
        model=ScriptedChatModel(script),
        wiki_store=RevisionWikiStore(revisions, "toy"),
        skill_store=RevisionSkillStore(revisions, "toy"),
        trace_store=RevisionTraceStore(revisions, "toy"),
        trace_source=ToyTraces(),
        evaluator=ToyEvaluator(),
        # The paper's gate accepts any increase of the mean. PairedGate accepts only when the paired bootstrap over the
        # validation tasks agrees; the toy split is tiny, so the thresholds are relaxed to make that visible.
        gate=PairedGate(min_tasks=len(VALIDATION), min_win_probability=0.8),
    )
    export_workspace(root, wiki=report.wiki, skills=report.skills)
    print(json.dumps(report.to_dict(), indent=2))
    print((Path(root) / "wiki" / "skill-impact.md").read_text())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "./toy-workspace")
