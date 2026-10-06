"""The comparator's own failure, from the library call to the rendered sentence.

`math-verify` bounds every comparison and scores the answer wrong when the bound
is hit. That is the conservative rule and these tests do not touch it. What they
pin is that the rule is now *visible*: a timeout was previously an unattributed
line on stderr and an ordinary `False` everywhere else, so a correct answer could
be marked incorrect and no artifact of the run would say so.

The stop rule runs the other way too. A run in which nothing timed out has to
produce exactly the report it produced before any of this existed -- the frozen
fixture in `test_report.py` is the half of that guarantee that would fail loudly,
and `test_a_run_without_timeouts_says_nothing_about_them` is the half that says
why the fixture still passes.
"""

from __future__ import annotations

import json

import pytest

from limite_evals import suites
from limite_evals.aggregate import comparison_timeout_rate, summarize, summarize_task
from limite_evals.report import render_markdown
from limite_evals_core import ladder
from limite_evals_core.equivalence import Comparison, compare, equivalent
from limite_evals_core.ladder import score
from limite_evals_core.schema import (
    Fingerprint,
    Pins,
    RunManifest,
    RunSummary,
    SampleResult,
    Sampling,
)

#: Declared here rather than imported from `test_report.py`, so this file
#: exercises the reporting seam without a second test module having to stay
#: shaped the way this one reads it.
MANIFEST = RunManifest(
    run_id="r001",
    created_at="2026-08-07T00:00:00Z",
    checkpoint="/work/ckpt",
    stage="pretrain",
    profile="base-kshot",
    suite="math-standard",
    sampling=Sampling(temperature=0.6, top_p=0.95, max_tokens=4096, skip_special_tokens=True),
    pins=Pins(
        verifiers="d30a3f48",
        research_environments="f9c43a74",
        math_verify="0.9.0",
        dataset_revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    ),
    fingerprint=Fingerprint(
        architecture="LimiteForCausalLM",
        artifact_sha256="a" * 64,
        template_sha256="b" * 64,
        max_model_len=8192,
        dtype="bfloat16",
    ),
)

#: A comparison the pinned `math-verify` cannot finish. `9^{9^{9^{9}}}` parses
#: without complaint and then asks sympy for an integer with more digits than
#: there are atoms, so the bound is hit in the compare step rather than the parse
#: one -- which is the step the observed run's 31 warnings came from.
#:
#: Synthetic rather than a row from OlympiadBench, deliberately. The pathological
#: golds found by the earlier isolated measurement compare in under a fifth of a
#: second here; the timeouts in `preflight-003` came from AIME and HMMT
#: *predictions*, so pinning this to a dataset row would pin the wrong thing and
#: would break whenever a revision moved.
UNFINISHABLE = "9^{9^{9^{9}}}"

#: Short enough to keep the two tests that use the real library near a second
#: each, long enough that a loaded machine does not hit it by accident. The
#: production bound is `equivalence.TIMEOUT`; what is under test is the
#: mechanism, which is the same at either value.
BOUND = 1


def _timing_out(_gold: str, _candidate: str) -> Comparison:
    """`compare`, as it behaves when the bound is hit: a mismatch, and a timeout."""
    return Comparison(equal=False, timed_out=True)


def _sample(problem: int, rollout: int, **overrides) -> SampleResult:
    fields = {
        "task": "aime25",
        "problem_index": problem,
        "rollout_index": rollout,
        "gold": "42",
        "completion": "\\boxed{42}",
        "strict": True,
        "lenient": True,
        "permissive": True,
        "format_ok": True,
    }
    return SampleResult(**{**fields, **overrides})


# --- What the library actually does -----------------------------------------


def test_math_verify_reports_a_timeout_as_an_ordinary_mismatch() -> None:
    """The bug, stated as a test: `False` from a timeout and `False` from a real
    mismatch are the same `False`.

    `equivalent` is the whole of what the ladder could see before this change,
    and it cannot tell the two apart. `compare` runs the identical call and
    returns the fact beside the verdict.
    """
    assert equivalent("1", UNFINISHABLE, timeout=BOUND) is False
    assert compare("1", "2", timeout=BOUND) == Comparison(equal=False, timed_out=False)

    timed_out = compare("1", UNFINISHABLE, timeout=BOUND)
    assert timed_out.equal is False
    assert timed_out.timed_out is True


def test_the_wrapper_could_not_have_caught_the_timeout_itself() -> None:
    """Why the detection watches a logger instead of catching something.

    `math-verify` catches its own `TimeoutException` inside `verify`, warns, and
    returns `False`, so nothing reaches this module to be caught. Even if it
    did, `TimeoutException` derives from `BaseException`, so the `except
    Exception` that scores every other failure 0.0 would let it through and kill
    the rollout.
    """
    from math_verify.errors import TimeoutException

    assert issubclass(TimeoutException, BaseException)
    assert not issubclass(TimeoutException, Exception)


def test_watching_the_logger_leaves_no_handler_behind() -> None:
    """The detection observes and then puts everything back.

    A handler or a raised level left on `math_verify` would follow the process
    out of the comparison and change where an operator's own logging goes.
    """
    import logging

    logger = logging.getLogger("math_verify")
    before = (list(logger.handlers), logger.level)
    compare("1", UNFINISHABLE, timeout=BOUND)
    assert (list(logger.handlers), logger.level) == before


# --- What the ladder does with it -------------------------------------------


def test_a_timeout_reaches_the_ladder_from_any_rung(monkeypatch) -> None:
    monkeypatch.setattr(ladder, "compare", _timing_out)
    assert score("\\boxed{42}", "42").comparison_timeout is True
    assert score("the answer is 42", "42").comparison_timeout is True
    assert score("#### 42", "42", answer_format="hashed").comparison_timeout is True


def test_a_timeout_still_scores_wrong(monkeypatch) -> None:
    """The constraint this change is built under.

    Making a timeout score anything other than wrong would re-baseline every
    number this repository has already reported. A completion whose comparison
    gave up fails every rung exactly as it did before, and the only difference
    is that the run now says so.
    """
    monkeypatch.setattr(ladder, "compare", _timing_out)
    result = score("\\boxed{42}", "42")
    assert (result.strict, result.lenient, result.permissive) == (False, False, False)
    # And the format is still recorded for what it was: the answer was boxed.
    assert result.format_ok is True


def test_a_completion_nothing_timed_out_on_says_so() -> None:
    assert score("\\boxed{42}", "42").comparison_timeout is False
    assert score("\\boxed{41}", "42").comparison_timeout is False
    assert score("no answer anywhere", "42").comparison_timeout is False


def test_the_rungs_keep_short_circuiting(monkeypatch) -> None:
    """A timeout is collected along the path the rungs already walk, not by
    comparing eagerly to make it easy to collect.

    `strict` matching means `lenient` and the 64 permissive candidates are never
    compared, and that has to stay true: scoring them anyway to gather timeouts
    would multiply `math-verify` calls per rollout for numbers nobody reads.
    """
    calls = []

    def counting(gold: str, candidate: str) -> Comparison:
        calls.append(candidate)
        return Comparison(equal=True, timed_out=False)

    monkeypatch.setattr(ladder, "compare", counting)
    assert score("\\boxed{42}", "42").strict is True
    assert calls == ["42"]


# --- What a run records and reports -----------------------------------------


def test_the_rate_is_the_fraction_of_rollouts_that_timed_out() -> None:
    samples = [
        _sample(problem, rollout, comparison_timeout=(problem == 0))
        for problem in range(4)
        for rollout in range(2)
    ]
    assert comparison_timeout_rate(samples) == 0.25
    assert summarize_task("aime25", samples, resamples=50).comparison_timeout_rate == 0.25


def test_a_taskset_that_never_timed_out_reports_zero() -> None:
    samples = [_sample(problem, 0) for problem in range(4)]
    assert comparison_timeout_rate(samples) == 0.0
    assert summarize_task("aime25", samples, resamples=50).comparison_timeout_rate == 0.0
    assert comparison_timeout_rate([]) == 0.0


def test_the_rate_is_not_a_declared_metric() -> None:
    """It qualifies the metrics rather than joining them.

    A member of the metric set becomes a table row, a plotted bar and a clause
    in the report's explanatory paragraph. This is a statement *about* those
    numbers -- that they are a lower bound -- so it travels beside them instead,
    which is also what keeps a zero-timeout run rendering unchanged.
    """
    summary = summarize_task("aime25", [_sample(0, 0, comparison_timeout=True)], resamples=50)
    assert "comparison_timeout" not in summary.metrics
    assert "comparison_timeout_rate" not in summary.metrics
    assert "comparison_timeout" not in summary.metric_set.metrics
    # And it is not something a truncated completion is forced to fail, either:
    # it says what the verifier did, not what the completion was.
    assert "comparison_timeout" not in summary.metric_set.correctness


def test_the_report_gives_the_rate_and_calls_the_rungs_a_lower_bound() -> None:
    samples = [
        _sample(problem, rollout, comparison_timeout=(problem == 0))
        for problem in range(4)
        for rollout in range(2)
    ]
    markdown = render_markdown(summarize(MANIFEST, samples, resamples=50))

    assert "The answer comparator timed out on **25.0%** of rollouts." in markdown
    assert "lower bound" in markdown
    # Directly under the table it qualifies, against the `truncated` row.
    body = markdown.split("## aime25")[1]
    assert body.index("| `truncated` |") < body.index("The answer comparator timed out")


def test_a_run_without_timeouts_says_nothing_about_them() -> None:
    """The other half of `test_report.py`'s frozen fixture, said out loud.

    Silence has always meant "no timeouts" in these reports. The difference is
    that it now means it because they were counted rather than because nobody
    looked.
    """
    markdown = render_markdown(
        summarize(MANIFEST, [_sample(problem, 0) for problem in range(4)], resamples=50)
    )
    assert "comparator" not in markdown
    assert "timed out" not in markdown


def test_one_timeout_in_a_large_taskset_is_not_rendered_as_none() -> None:
    """A single timeout in MMLU-Pro's 12,032 rollouts is 0.008%.

    At the one decimal every other percentage in the report uses that prints as
    `0.0%`, inside a sentence stating that it happened.
    """
    samples = [_sample(problem, 0, comparison_timeout=(problem == 0)) for problem in range(12032)]
    markdown = render_markdown(summarize(MANIFEST, samples, resamples=10))
    assert "timed out on **0.0083%** of rollouts" in markdown


# --- The seams a result file has to survive ---------------------------------


def test_a_result_file_written_before_the_field_existed_still_loads() -> None:
    """Both files a run leaves behind: the rollouts and the numbers.

    A rollout with no `comparison_timeout` is a rollout nothing timed out on,
    and a taskset with no rate is a taskset with a rate of zero. Any other
    default would invent timeouts in every run already on disk.
    """
    row = {
        "task": "aime25",
        "problem_index": 0,
        "rollout_index": 0,
        "gold": "42",
        "completion": "\\boxed{42}",
        "truncated": False,
        "strict": True,
        "lenient": True,
        "permissive": True,
        "format_ok": True,
    }
    assert SampleResult.model_validate(row).comparison_timeout is False

    summary = summarize(MANIFEST, [_sample(0, 0)], resamples=10)
    reloaded = json.loads(summary.model_dump_json())
    del reloaded["tasks"][0]["comparison_timeout_rate"]
    assert RunSummary.model_validate(reloaded).tasks[0].comparison_timeout_rate == 0.0


async def test_the_runner_records_the_timeout_on_the_rollout(monkeypatch) -> None:
    """End to end: a comparison that gives up reaches `samples.jsonl`."""
    import httpx

    from limite_evals import profiles, runner, tasksets, templates

    monkeypatch.setattr(ladder, "compare", _timing_out)
    spec = suites.taskset("math-extended", "math500")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "\\boxed{42}"}, "finish_reason": "stop"}]},
        )

    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient

    class Patched(original):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs):
            super().__init__(transport=transport, **kwargs)

    httpx.AsyncClient = Patched  # type: ignore[misc]
    try:
        results = await runner.run_taskset(
            spec,
            [("Q", "42")],
            base_url="http://engine",
            model="limite",
            profile=profiles.resolve("chat"),
            # Rendered raw rather than through the profile's own template: `chat`
            # serves the artifact's, and there is no artifact here. What this test
            # asserts is what `compare` did, which no rendering changes.
            rendering=templates.resolve(taskset=spec.name, template="base-kshot-math", stop=[]),
            bindings=tasksets.taskset_protocols(spec),
            max_tokens=128,
        )
    finally:
        httpx.AsyncClient = original  # type: ignore[misc]

    assert all(result.comparison_timeout for result in results)
    assert not any(result.strict for result in results)
    assert '"comparison_timeout":true' in results[0].model_dump_json().replace(" ", "")
