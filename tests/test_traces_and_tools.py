# Copyright 2026 Vikash Ranjan (CTO, styls.ai). Licensed under the Apache License, Version 2.0.
import pytest

from skillwiki.model import ImagePart
from skillwiki.tools import Scratchpad, ToolCall, ToolCallError, parse_tool_call
from skillwiki.traces import TaskOutcome, Trace, stratified_sample, summarize_outcomes


def _trace(i, passed):
    return Trace(id=f"t{i}", outcome=TaskOutcome(task_id=f"task{i}", score=1.0 if passed else 0.0, passed=passed, prediction="p", truth="y"), text=f"body {i}")


def test_stratified_sample_caps_and_rotates_without_replacement():
    traces = [_trace(i, passed=i % 2 == 0) for i in range(20)]  # 10 failing, 10 passing
    sample, seen = stratified_sample(traces, failing=5, passing=3, seen=set(), iteration=1)
    assert sum(not t.outcome.passed for t in sample) == 5 and sum(t.outcome.passed for t in sample) == 3
    sample2, seen2 = stratified_sample(traces, failing=5, passing=3, seen=seen, iteration=2)
    assert not ({t.id for t in sample} & {t.id for t in sample2})  # second iteration sees fresh evidence
    # third iteration: failing stratum exhausted (10 seen), so it resets rather than sampling fewer than 5
    sample3, _ = stratified_sample(traces, failing=5, passing=3, seen=seen2, iteration=3)
    assert sum(not t.outcome.passed for t in sample3) == 5
    # deterministic for a given seed and iteration
    assert [t.id for t in stratified_sample(traces, seen=set(), iteration=1)[0]] == [t.id for t in sample]


def test_summary_and_render():
    traces = [_trace(1, False), _trace(2, True)]
    summary = summarize_outcomes(traces)
    assert "1/2 passed" in summary and "- t1 [FAIL]" in summary and "prediction='p' truth='y'" in summary
    long = Trace(id="x", outcome=TaskOutcome("k", 0.0, False), text="a" * 100)
    rendered = long.rendered(char_cap=10)
    assert "truncated 90 characters" in rendered and "FAIL" in rendered


def test_parse_tool_call_variants():
    assert parse_tool_call('{"tool": "read_index", "args": {}}') == ToolCall("read_index", {})
    assert parse_tool_call('Let me look.\n```json\n{"tool":"read_trace","args":{"id":"t1"}}\n```') == ToolCall("read_trace", {"id": "t1"})
    assert parse_tool_call('{"tool": "read_pattern", "name": "loop"}').args == {"name": "loop"}
    with pytest.raises(ToolCallError, match="no tool call"):
        parse_tool_call("thinking out loud {\"not\": 1}")
    with pytest.raises(ToolCallError, match="exactly one"):
        parse_tool_call('{"tool": "a"} {"tool": "b"}')
    with pytest.raises(ToolCallError, match="unknown tool"):
        parse_tool_call('{"tool": "rm"}', allowed={"read_index"})


def test_scratchpad_folds_and_attaches_only_latest_media():
    pad = Scratchpad()
    image = ImagePart("image/png", "AAAA", ref="img1")
    pad.add(ToolCall("read_trace", {"id": "t1", "include_media": True}), "", "first observation line\nmore " * 50, [image])
    pad.add(ToolCall("read_index", {}), "", "index")
    text = pad.render(char_cap=100_000)
    assert "shown at that turn, no longer attached" in text and "img1" in text
    parts = pad.parts(char_cap=100_000)
    assert not any(isinstance(p, ImagePart) for p in parts)  # media only travel in the turn that fetched them
    pad.add(ToolCall("read_trace", {"id": "t2", "include_media": True}), "", "obs", [image])
    assert any(isinstance(p, ImagePart) for p in pad.parts(char_cap=100_000))
    folded = pad.render(char_cap=400)
    assert "Observation (folded): first observation line" in folded and len(folded) <= 400
