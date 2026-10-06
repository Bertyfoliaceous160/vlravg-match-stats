"""OlympiadBench's MathArena cutover and durable scoring identity."""

from __future__ import annotations

import pytest

from limite_evals_core.protocols import (
    TASKSET_EXACT,
    Grader,
    PublishedGrader,
    exact_grader,
    execute_published,
)

# These are engine outcomes, including retained upstream defects, not claims
# that every accepted answer is mathematically correct.
MATHARENA_VECTORS = [
    (r"\boxed{166}", "$166$", True),
    (r"\boxed{\frac{41}{12}}", r"$\frac{41}{12}$", True),
    (r"\boxed{x^2=1}", "$x^2=1$", True),
    (r"\boxed{2,3}", "$2,3$", True),
    (r"\boxed{3,2}", "$2,3$", True),
    (r"\boxed{2}", "$2,3$", False),
    (r"\boxed{(2,3)}", "$(2,3)$", True),
    (r"\boxed{[0,1]}", "$[0,1]$", False),
    (r"\boxed{0}", "$0$", False),
    (r"Work: \boxed{1}. Finally \boxed{277}", "$277$", True),
    ("So the final answer is\n" + r"\boxed{277}", "$277$", True),
    ("The answer is 277", "$277$", True),
    (r"\fbox{277}", "$277$", True),
    (r"\boxed{277", "$277$", True),
    (r"<think>\boxed{277}</think>", "$277$", True),
    (r"<think>\boxed{277}", "$277$", True),
    (r"\boxed{4.2}", "$420$", False),
    (r"\boxed{420}", "$4.2$", False),
    (r"\boxed{278}", "$277$", False),
]


@pytest.mark.parametrize(("completion", "gold", "expected"), MATHARENA_VECTORS)
def test_olympiadbench_executes_the_same_matharena_engine_as_aime(
    completion: str, gold: str, expected: bool
) -> None:
    declared = TASKSET_EXACT["aime26"]
    assert isinstance(declared, PublishedGrader)
    namespace = execute_published(declared.sources)
    _, correct, _ = namespace["extract_and_grade"](
        [{"role": "assistant", "content": completion}], 0, gold, {"strict_parsing": False}
    )
    assert bool(correct) is expected
    grader = exact_grader("olympiadbench")
    assert isinstance(grader, Grader)
    assert grader.grade(completion, gold) is expected
    assert "MathArena applied to OlympiadBench by Limite" in grader.name


def test_only_olympiadbench_gets_a_new_answer_scorer_pin() -> None:
    from limite_evals_core.answer_scoring import answer_scorer_revisions

    assert answer_scorer_revisions(["aime25", "aime26", "hmmt25"]) == ""
    pin = answer_scorer_revisions(["olympiadbench"])
    assert pin.startswith("olympiadbench=")
    assert "a11194deff8c67a232974a383795e8a2776b4c6f" in pin
    assert answer_scorer_revisions(["aime26", "olympiadbench"]) == pin


def test_old_and_new_olympiadbench_results_form_separate_comparison_groups() -> None:
    from test_collect import manifest

    from limite_evals.collect import comparison_groups
    from limite_evals_core.answer_scoring import answer_scorer_revisions

    old = manifest()
    new = old.model_copy(
        update={
            "pins": old.pins.model_copy(
                update={"answer_scorer_revision": answer_scorer_revisions(["olympiadbench"])}
            )
        }
    )
    assert any("answer scorer revision" in reason for reason in old.comparable_with(new))
    assert comparison_groups([old, new]) == [0, 1]
    assert old.comparable_with(old.model_copy(update={"scorer_version": "older"})) == []


def test_aggregation_rejects_cached_openbmb_verdicts_under_the_new_pin() -> None:
    from test_rescore import MANIFEST, _sample

    from limite_evals.aggregate import summarize
    from limite_evals_core.answer_scoring import answer_scorer_revisions
    from limite_evals_core.schema import ProtocolSample

    sample = _sample("olympiadbench", 0, completion=r"\boxed{420}", gold="$4.2$")
    sample.protocols = {
        "exact": ProtocolSample(
            protocol="exact", anchor="OpenBMB/OlympiadBench historical scorer", correct=True
        )
    }
    revised = MANIFEST.model_copy(
        update={
            "pins": MANIFEST.pins.model_copy(
                update={"answer_scorer_revision": answer_scorer_revisions(["olympiadbench"])}
            )
        }
    )
    with pytest.raises(ValueError, match="answer scorer provenance"):
        summarize(revised, [sample])
    # Reading and aggregating the old regime retains its blank, historical pin.
    historical = summarize(MANIFEST, [sample])
    assert historical.manifest.pins.answer_scorer_revision == ""
    assert historical.tasks[0].protocol("exact").anchor == sample.protocols["exact"].anchor


async def test_runner_rescore_report_and_collection_keep_the_matharena_identity(
    tmp_path, monkeypatch
) -> None:
    import httpx
    from test_rescore import MANIFEST

    from limite_evals import (
        aggregate,
        cli,
        collect,
        profiles,
        report,
        rescore,
        runner,
        suites,
        tasksets,
        templates,
    )
    from limite_evals_core.answer_scoring import answer_scorer_revisions

    spec = suites.taskset("math-extended", "olympiadbench")
    completion = r"Scratch \boxed{1}. Finally \boxed{277}"
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": completion}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 20},
            },
        )
    )
    original = httpx.AsyncClient

    def client(**kwargs):
        return original(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    samples = await runner.run_taskset(
        spec,
        [("Q", "$277$")],
        base_url="http://engine",
        model="fixture",
        profile=profiles.resolve("chat"),
        rendering=templates.resolve(taskset=spec.name, template="base-kshot-math", stop=[]),
        bindings=tasksets.taskset_protocols(spec),
        max_tokens=128,
    )
    rescored_sample = rescore.rescore_sample(spec, samples[0])
    assert rescored_sample.scorer_version != "2026.09-native-physics"
    assert samples[0].scorer_version == rescored_sample.scorer_version
    assert samples[0].protocols == rescored_sample.protocols
    assert samples[0].protocols["exact"].correct is True
    assert samples[0].protocols["lenient"].relaxes == "exact"
    assert samples[0].protocols["permissive"].relaxes == "exact"

    manifest = MANIFEST.model_copy(
        update={
            "suite": "math-extended",
            "relaxation_bases": cli._relaxation_bases((spec,)),
            "pins": MANIFEST.pins.model_copy(
                update={"answer_scorer_revision": answer_scorer_revisions([spec.name])}
            ),
        }
    )
    summary = aggregate.summarize(manifest, samples, metric_sets=cli._metric_sets((spec,)))
    run_dir = report.write_run(summary, samples, root=tmp_path)
    new_summary = rescore.rescore_run(run_dir, tmp_path / "rescored", workers=1)
    assert new_summary.manifest.pins.answer_scorer_revision == manifest.pins.answer_scorer_revision
    assert new_summary.tasks[0].metrics["exact"] == summary.tasks[0].metrics["exact"]
    assert "MathArena applied to OlympiadBench by Limite" in report.render_markdown(new_summary)
    entry = collect.run_entry(run_dir, new_summary, True)["tasksets"][0]
    assert "MathArena" in entry["metrics"]["exact"]["anchor"]
    assert "MathArena" in entry["metrics"]["pass@4"]["anchor"]
    for name in ("lenient", "permissive"):
        assert "MathArena" in entry["metrics"][name]["provenance"]
        row = next(
            line
            for line in report.render_markdown(new_summary).splitlines()
            if line.startswith(f"| `{name}` |")
        )
        assert "MathArena" in row


def test_rescore_recomputes_historical_openbmb_verdicts_and_preserves_the_source(tmp_path) -> None:
    from test_rescore import MANIFEST, _sample

    from limite_evals import aggregate, cli, report, rescore, suites
    from limite_evals_core.answer_scoring import answer_scorer_revisions
    from limite_evals_core.schema import ProtocolSample

    spec = suites.taskset("math-extended", "olympiadbench")
    sample = _sample(
        "olympiadbench",
        0,
        gold="$4.2$",
        completion=r"\boxed{420}",
        metrics={"exact": 1.0},
        scorer_version="2026.09-native-physics",
    )
    sample.protocols = {
        "exact": ProtocolSample(
            protocol="exact", anchor="OpenBMB/OlympiadBench historical scorer", correct=True
        )
    }
    legacy = aggregate.summarize(
        MANIFEST.model_copy(update={"suite": "math-extended"}),
        [sample],
        metric_sets=cli._metric_sets((spec,)),
    )
    source = report.write_run(legacy, [sample], root=tmp_path)
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    revised = rescore.rescore_run(source, tmp_path / "new", workers=1)
    assert revised.tasks[0].metrics["exact"].value == 0
    assert revised.manifest.pins.answer_scorer_revision == answer_scorer_revisions([spec.name])
    assert {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()} == before
    with pytest.raises(FileExistsError, match="already exists"):
        rescore.rescore_run(source, source, workers=1)


def test_olympiadbench_matharena_gold_characterization() -> None:
    from limite_evals import suites, tasksets

    spec = suites.taskset("math-extended", "olympiadbench")
    try:
        rows = tasksets.load_raw_rows(spec)
    except Exception as exc:  # Dataset availability is opportunistic, like the reference parity test.
        pytest.skip(f"OlympiadBench unavailable at the pinned revision: {exc}")
    assert len(rows) == 674
    grader = exact_grader(spec.name)
    assert isinstance(grader, Grader)
    failures = set()
    for row in rows:
        gold = tasksets.render_row(spec, row)[1]
        body = gold.strip()
        if body.startswith("$") and body.endswith("$"):
            body = body[1:-1].strip()
        if not grader.grade(f"So the final answer is \\boxed{{{body}}}.", gold):
            failures.add(row["id"])
    assert failures == {
        2560,
        2950,
        1929,
        1930,
        2313,
        2570,
        2317,
        2318,
        1807,
        2962,
        1940,
        1687,
        2330,
        2203,
        2332,
        2460,
        2336,
        2720,
        2341,
        2472,
        1962,
        1970,
        2486,
        2235,
        3004,
        2238,
        2498,
        2502,
        2376,
        1737,
        2255,
        1618,
        2523,
        2276,
        1766,
        2024,
        2410,
        1645,
        2544,
        2292,
        2298,
        2427,
        2303,
    }


def test_summary_assembly_preserves_the_scorer_pin_and_rejects_stale_verdicts() -> None:
    from test_rescore import MANIFEST, _sample

    from limite_evals import aggregate, cli, collect, rescore, suites
    from limite_evals_core.answer_scoring import answer_scorer_revisions

    fragments = []
    for name in ("aime26", "olympiadbench"):
        spec = suites.taskset("math-extended", name)
        sample = rescore.rescore_sample(spec, _sample(name, 0, gold="42", completion=r"\boxed{42}"))
        manifest = MANIFEST.model_copy(
            update={
                "run_id": name,
                "suite": "math-extended",
                "relaxation_bases": cli._relaxation_bases((spec,)),
                "pins": MANIFEST.pins.model_copy(
                    update={
                        "dataset_revision": cli._dataset_revisions((spec,)),
                        "answer_scorer_revision": answer_scorer_revisions([name]),
                    }
                ),
            }
        )
        fragments.append(aggregate.summarize(manifest, [sample], metric_sets=cli._metric_sets((spec,))))
    merged = collect.merge_summaries(fragments)
    assert merged.manifest.pins.answer_scorer_revision == answer_scorer_revisions(["olympiadbench"])
    assert merged.tasks[1].protocol_source("lenient") == merged.tasks[1].protocol("exact").anchor

    olympiad = fragments[1].model_copy(deep=True)
    olympiad.tasks[0].protocols = [
        record.model_copy(update={"anchor": "OpenBMB/OlympiadBench historical scorer"})
        if record.protocol == "exact"
        else record
        for record in olympiad.tasks[0].protocols
    ]
    with pytest.raises(ValueError, match="answer scorer provenance"):
        collect.merge_summaries([fragments[0], olympiad])
