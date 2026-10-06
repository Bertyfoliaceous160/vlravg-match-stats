"""The `math-extended` suite: what is transcribed, what is chosen, what is frozen.

Four claims are asserted here rather than argued in prose, because each one is
the kind that fails silently:

* **Each `math-standard` prompt is pinned.** GSM8K deliberately reuses the
  MATH-500 boxed instruction; the other three remain unchanged.
* **AIME 2026 is transcription, not authorship.** Its instruction is compared
  byte-for-byte against the constant already in `suites.py`; "we read them and
  they looked the same" is not evidence.
* **OlympiadBench's instruction really does vary per row.** The variant asking for
  several answers is the whole reason the spec learned to take a callable, and 93
  of the 674 rows need it.
* **BeyondAIME extends rather than changes the suite prefix.** It is appended as
  the eighth taskset while `math-standard` remains byte-for-byte frozen.

The parity vectors follow `tests/test_strict_parity.py`, which is the file this
repository's comparability argument rests on and which must not be relaxed. This
one adds vectors drawn from the *new* tasksets' real gold answers; it does not
alter the rules that file pins.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download
from verifiers.v1.utils.score import verify_boxed_math_answer

from limite_evals import profiles, suites, tasksets, templates
from limite_evals.tasksets import load_raw_rows, render_row
from limite_evals_core.ladder import score

# --- The frozen suite ---------------------------------------------------------

#: Every field of every `math-standard` spec, captured before `math-extended`
#: existed. Comparing whole dataclasses rather than a chosen few fields is
#: deliberate: a subset assertion passes for exactly the drift it fails to name.
MATH_STANDARD_BEFORE = {
    "aime24": {
        "verifiers_id": "aime24-v1",
        "dataset": "HuggingFaceH4/aime_2024",
        "split": "train",
        "revision": "2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
        "group_size": 32,
        "instruction_position": "prepend",
        "answer_format": "boxed",
        "dataset_config": None,
        "dataset_subsets": (),
        "answer_list_field": None,
        "licence": None,
        "caveat": None,
    },
    "aime25": {
        "verifiers_id": "aime25-v1",
        "dataset": "opencompass/AIME2025",
        "split": "test",
        "revision": "a6ad95f611d72cf628a80b58bd0432ef6638f958",
        "group_size": 32,
        "instruction_position": "prepend",
        "answer_format": "boxed",
        "dataset_config": None,
        "dataset_subsets": ("AIME2025-I", "AIME2025-II"),
        "answer_list_field": None,
        "licence": None,
        "caveat": None,
    },
    "math500": {
        "verifiers_id": "math500-v1",
        "dataset": "HuggingFaceH4/MATH-500",
        "split": "test",
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "group_size": 4,
        "instruction_position": "append",
        "answer_format": "boxed",
        "dataset_config": None,
        "dataset_subsets": (),
        "answer_list_field": None,
        "licence": None,
        "caveat": None,
    },
    "gsm8k": {
        "verifiers_id": "gsm8k-v1",
        "dataset": "openai/gsm8k",
        "split": "test",
        "revision": "",
        "group_size": 1,
        "instruction_position": "append",
        "answer_format": "boxed",
        "dataset_config": "main",
        "dataset_subsets": (),
        "answer_list_field": None,
        "licence": None,
        "caveat": None,
    },
}


def test_math_standard_spec_fields_match_the_current_contract() -> None:
    """The non-prompt taskset contract, made mechanical.

    `instruction` is checked separately below because it is compared against the
    module constants rather than an inlined copy of their text.
    """
    actual = {
        spec.name: {
            field: getattr(spec, field)
            for field in MATH_STANDARD_BEFORE[spec.name]
        }
        for spec in suites.MATH_STANDARD
    }
    assert actual == MATH_STANDARD_BEFORE


def test_math_standard_prompt_components_are_declared() -> None:
    instructions = {spec.name: spec.instruction for spec in suites.MATH_STANDARD}
    assert instructions == {
        "aime24": suites.AIME_INSTRUCTION,
        "aime25": suites.AIME_INSTRUCTION,
        "math500": suites.MATH500_INSTRUCTION,
        "gsm8k": suites.MATH500_INSTRUCTION,
    }


def test_math_standard_rendered_prompts_match_their_sources() -> None:
    """The prompt join is a source-owned protocol, not an incidental detail."""
    probe = "Find $n$ such that \\(n^2 = 4\\).\nShow work."
    rendered = {spec.name: spec.render_problem(probe) for spec in suites.MATH_STANDARD}
    assert rendered == {
        "aime24": suites.AIME_INSTRUCTION + probe,
        "aime25": suites.AIME_INSTRUCTION + probe,
        "math500": probe + suites.MATH500_INSTRUCTION,
        "gsm8k": probe + suites.MATH500_INSTRUCTION,
    }


def test_math_standard_is_still_resolvable_and_first_in_math_extended() -> None:
    """`math-extended` extends rather than reshuffles: the four existing tasksets
    are the same objects, in the same order, followed by the four additions."""
    standard = suites.MATH_STANDARD
    extended = suites.resolve("math-extended")
    assert extended[: len(standard)] == standard
    assert [spec.name for spec in extended[len(standard) :]] == [
        "aime26",
        "hmmt25",
        "olympiadbench",
        "beyondaime",
        "apex-shortlist",
        "hmmt26",
    ]


# --- AIME 2026: transcription -------------------------------------------------


def test_aime26_instruction_is_byte_identical_to_the_existing_constant() -> None:
    """The claim that AIME 2026 is transcribed, not authored.

    `aime26_v1/taskset.py` at `f9c43a74` defines its own `INSTRUCTION`, and it is
    the same string `aime24_v1` and `aime25_v1` use -- which is why this taskset
    needs no new constant. Asserting equality against the existing one is what
    turns "we compared them by eye" into something that fails if either moves.
    """
    aime26 = suites.taskset("math-extended", "aime26")
    reference = (
        "Solve the following math problem. Explain your reasoning and put the "
        "final answer in \\boxed{}.\n\n"
    )
    assert aime26.instruction == reference
    assert aime26.instruction == suites.AIME_INSTRUCTION
    assert aime26.instruction is suites.AIME_INSTRUCTION


def test_aime26_transcribes_its_reference_fields() -> None:
    """Dataset, split and revision are read off the reference taskset; the budget
    mirrors the existing AIME entries under decision 11."""
    aime26 = suites.taskset("math-extended", "aime26")
    aime24 = suites.taskset("math-extended", "aime24")
    assert aime26.dataset == "MathArena/aime_2026"
    assert aime26.split == "train"
    assert aime26.revision == "10b4e45b7a503075d4da8a0d57916a4f06ce6bd2"
    assert aime26.answer_format == "boxed"
    assert aime26.group_size == aime24.group_size


# --- HMMT: declared choices ---------------------------------------------------


def test_hmmt25_pins_its_revision_and_group_size() -> None:
    """The group size is a choice, not a transcription -- HMMT has no reference
    environment at the pin. The assertion exists so the choice cannot drift
    unnoticed into looking like a transcribed value. There is no budget to pin:
    the generation budget is native-max from the served model."""
    hmmt = suites.taskset("math-extended", "hmmt25")
    assert hmmt.dataset == "MathArena/hmmt_feb_2025"
    assert hmmt.split == "train"
    assert hmmt.revision == "6fdc4277120810ff75aa22d2d5489b91f7a262a1"
    assert hmmt.group_size == 32


def test_hmmt26_pins_its_revision_and_group_size() -> None:
    """The same declared choice as `hmmt25` -- there is no reference environment
    for February 2026 either, so the instruction's provenance is a sibling
    taskset."""
    hmmt = suites.taskset("math-extended", "hmmt26")
    assert hmmt.dataset == "MathArena/hmmt_feb_2026"
    assert hmmt.split == "train"
    assert hmmt.revision == "02fba4f74d8e68e73e66a02d540fd979c05c274c"
    assert hmmt.group_size == 32
    assert hmmt.instruction is suites.AIME_INSTRUCTION
    assert hmmt.instruction_position == "prepend"
    assert hmmt.answer_format == "boxed"


def test_apex_shortlist_pins_its_revision_and_group_size() -> None:
    """Same shape and same declared choices as the HMMT entries. MathArena ships
    an `instruction` line in this benchmark's own config and this taskset
    deliberately does not adopt it -- the assertion is that the prompt is the
    sibling AIME string, so a future edit that quietly switched to MathArena's
    wording would re-baseline the number rather than pass unnoticed."""
    apex = suites.taskset("math-extended", "apex-shortlist")
    assert apex.dataset == "MathArena/apex-shortlist"
    assert apex.split == "train"
    assert apex.revision == "f3efdf224ef665f129ddaae37699f6098c65781b"
    assert apex.group_size == 32
    assert apex.instruction is suites.AIME_INSTRUCTION
    assert apex.instruction_position == "prepend"
    assert apex.answer_format == "boxed"


def test_the_non_commercial_licence_travels_with_the_taskset() -> None:
    """Decision 9 admitted CC BY-NC-SA 4.0 only on condition the licence stays
    visible beside any number, so it is spec data rather than a commit message."""
    extended = {spec.name: spec.licence for spec in suites.resolve("math-extended")}
    assert extended["hmmt25"] == "cc-by-nc-sa-4.0"
    assert extended["aime26"] == "cc-by-nc-sa-4.0"
    # The two MathArena additions arrive under the same licence and the same
    # condition; nothing about it is weakened by there being more of them.
    assert extended["hmmt26"] == "cc-by-nc-sa-4.0"
    assert extended["apex-shortlist"] == "cc-by-nc-sa-4.0"
    # And it did not leak onto the frozen four.
    assert {extended[name] for name in ("aime24", "aime25", "math500", "gsm8k")} == {None}


def test_the_contamination_caveat_travels_with_olympiadbench() -> None:
    """E-020 is an accepted risk, not a resolved one. The condition of accepting
    it was that it reaches every reader of a number, so it is carried as data."""
    olympiad = suites.taskset("math-extended", "olympiadbench")
    assert olympiad.caveat is not None
    assert "ontamination" in olympiad.caveat
    assert "February 2024" in olympiad.caveat
    for name in ("aime24", "aime25", "math500", "gsm8k", "aime26", "hmmt25", "hmmt26"):
        assert suites.taskset("math-extended", name).caveat is None


#: What each MathArena addition's pinned revision actually holds: the digest of
#: its single parquet shard, its row count, and the column set every row carries.
#:
#: The row count is the assertion that matters most, because it is the one the
#: published grading config also states -- `n_problems: 47` and `n_problems: 33`
#: -- so a revision that added or dropped a problem would leave the taskset and
#: the config it is graded under describing different benchmarks.
_MATHARENA_PINS = {
    "apex-shortlist": (
        "12f2300323d4d0cbd51b4c7528ae45b626e1681b0103c3df94d5488324c44acc",
        47,
        ("answer", "problem", "problem_idx", "source"),
    ),
    "hmmt26": (
        "e5fcff6b1c2262841c0c37bf6d7b42529f284528d5f0d5c45c52e8bf0654a916",
        33,
        ("answer", "problem", "problem_idx", "problem_type"),
    ),
}


@pytest.mark.parametrize("name", sorted(_MATHARENA_PINS))
def test_pinned_matharena_dataset_cardinality_schema_and_parquet_digest(name: str) -> None:
    """The pin is a claim about bytes, so it is checked against bytes.

    Skipped rather than failed when the Hub is unreachable, on the same rule the
    BeyondAIME equivalent uses: an offline checkout has no way to distinguish a
    moved revision from an absent network, and reporting the second as the first
    would be a false alarm that trains people to ignore it.
    """
    spec = suites.taskset("math-extended", name)
    digest, cardinality, columns = _MATHARENA_PINS[name]
    try:
        parquet = Path(
            hf_hub_download(
                repo_id=spec.dataset,
                filename="data/train-00000-of-00001.parquet",
                repo_type="dataset",
                revision=spec.revision,
                token=os.environ.get("HF_TOKEN") or None,
            )
        )
        rows = tasksets.load_raw_rows(spec)
    except Exception as exc:  # noqa: BLE001 - offline parity is an explicit skip
        pytest.skip(f"{name} not available at the pinned revision: {exc}")

    assert hashlib.sha256(parquet.read_bytes()).hexdigest() == digest
    assert len(rows) == cardinality
    assert {tuple(sorted(row)) for row in rows} == {columns}
    assert all(isinstance(row["problem"], str) and row["problem"] for row in rows)
    # Gold is a string on both, unlike BeyondAIME's integers: these are
    # competition answers that may be bare LaTeX, which is what the ladder's
    # equivalence is there to compare.
    assert all(isinstance(row["answer"], str) and row["answer"] for row in rows)


def test_the_apex_shortlist_caveat_names_problems_the_dataset_actually_carries() -> None:
    """The caveat is only worth carrying if the four problems it names are real.

    They are read off the shortlist's own `source` column rather than trusted:
    the whole reason the overlap is recorded as data is that a reader may want to
    subtract those rows, and a caveat naming problems the dataset does not label
    that way would send them looking for rows that are not there.
    """
    spec = suites.taskset("math-extended", "apex-shortlist")
    try:
        rows = tasksets.load_raw_rows(spec)
    except Exception as exc:  # noqa: BLE001 - offline parity is an explicit skip
        pytest.skip(f"apex-shortlist not available at the pinned revision: {exc}")

    sources = {row["source"] for row in rows}
    assert spec.caveat is not None
    for problem in ("AIME 2025 P14", "AIME 2025 P15", "HMMT Feb 2025 C9", "HMMT Feb 2025 C10"):
        assert problem in spec.caveat
        assert problem in sources


def test_the_overlap_caveat_travels_with_apex_shortlist() -> None:
    """The shortlist is drawn from other 2025 competitions and four of its 47
    problems are ones this suite already scores under `aime25` and `hmmt25`. The
    published set is kept whole -- trimming would make it a Limite-local benchmark
    -- so the double-count is carried the way OlympiadBench's contamination risk
    is, as data that reaches every reader of the number.

    The four are named rather than counted, because a reader who wants to subtract
    them needs to know which they are."""
    apex = suites.taskset("math-extended", "apex-shortlist")
    assert apex.caveat is not None
    for problem in ("AIME 2025 P14", "AIME 2025 P15", "HMMT Feb 2025 C9", "HMMT Feb 2025 C10"):
        assert problem in apex.caveat
    # Both tasksets it collides with are in this suite, which is what makes the
    # double-count real here rather than hypothetical.
    assert {"aime25", "hmmt25"} <= {spec.name for spec in suites.resolve("math-extended")}


# --- OlympiadBench: the per-row instruction -----------------------------------


def _olympiad_row(**overrides) -> dict:
    row = {"answer_type": "Numerical", "is_multiple_answer": False, "unit": None}
    return {**row, **overrides}


def test_olympiadbench_asks_for_several_answers_when_the_gold_holds_several() -> None:
    """The 14% this taskset exists to prevent.

    A fixed instruction would give all 674 rows the singular wording, so the 93
    rows whose gold holds multiple answers would be asked for one and scored
    against several -- a harness bug that reads as a model defect.
    """
    single = suites.olympiadbench_instruction(_olympiad_row(is_multiple_answer=False))
    multiple = suites.olympiadbench_instruction(_olympiad_row(is_multiple_answer=True))
    assert "The answer of The problem should be a numerical value." in single
    assert "The problem has multiple answers, each of them should be a numerical value." in multiple
    assert "\\boxed{answer}" in single
    assert "\\boxed{multiple answers connected with commas}" in multiple


def test_olympiadbench_names_each_answer_type_as_the_reference_does() -> None:
    """Verbatim from `english_answer_type_dict`. `Tuple` deliberately yields no
    sentence at all -- the reference's own note is that the word means different
    things in different problems and would mislead."""
    wording = {
        kind: suites.olympiadbench_instruction(_olympiad_row(answer_type=kind))
        for kind in ("Numerical", "Expression", "Equation", "Interval", "Tuple")
    }
    assert "should be a numerical value." in wording["Numerical"]
    assert "should be an expression." in wording["Expression"]
    assert "should be an equation." in wording["Equation"]
    assert "should be an interval." in wording["Interval"]
    assert "The answer of The problem should be" not in wording["Tuple"]
    assert "competition. Please calculate" in wording["Tuple"]


def test_olympiadbench_moves_the_unit_outside_the_box() -> None:
    """Transcribed because it changes what a correct completion looks like: the
    gold carries the bare value, so a unit inside `\\boxed{}` would not match."""
    united = suites.olympiadbench_instruction(_olympiad_row(unit="km"))
    assert "\\boxed{answer}(unit)" in united
    assert "note that the unit of the answer should not be included in \\boxed{}" in united
    assert "(unit)" not in suites.olympiadbench_instruction(_olympiad_row())


def test_olympiadbench_refuses_an_answer_type_it_cannot_word() -> None:
    """The reference calls `exit()` here. Raising is the one deliberate departure:
    the refusal to guess is kept, the killing of the interpreter is not."""
    with pytest.raises(ValueError, match="unrecognised OlympiadBench answer type"):
        suites.olympiadbench_instruction(_olympiad_row(answer_type="Graph"))


def test_olympiadbench_instruction_ends_with_the_reference_separator() -> None:
    """The reference joins prompt to question with an explicit newline, so the
    separator belongs to the instruction where `render_problem` concatenates."""
    instruction = suites.olympiadbench_instruction(_olympiad_row())
    assert instruction.endswith(".\n")
    spec = suites.taskset("math-extended", "olympiadbench")
    assert spec.instruction_position == "prepend"
    assert spec.render_problem("Q", _olympiad_row()) == instruction + "Q"


def test_olympiadbench_spec_reads_the_list_valued_gold() -> None:
    olympiad = suites.taskset("math-extended", "olympiadbench")
    assert olympiad.answer_list_field == "final_answer"
    assert olympiad.dataset_config == "OE_TO_maths_en_COMP"
    assert olympiad.revision == "91184b52131e7fc9455fef848035173aea8cc01a"
    assert olympiad.group_size == 4
    assert not isinstance(olympiad.instruction, str)


# --- Exemplar mapping ---------------------------------------------------------


def test_base_kshot_resolves_a_template_for_every_math_extended_taskset() -> None:
    """Without a `KSHOT_TEMPLATES` entry this raises before the engine starts, so
    the suite would be unrunnable under `base-kshot` rather than merely rendered
    oddly."""
    for spec in suites.resolve("math-extended"):
        rendering = templates.resolve(
            taskset=spec.name, template=templates.for_profile("base-kshot", spec.name)
        )
        assert rendering.template.startswith("{{ bos_token }}")


def test_the_new_tasksets_share_the_maths_exemplars() -> None:
    """As AIME 2024 and 2025 already do: competition maths with no train split of
    its own to draw exemplars from."""
    for name in ("aime26", "hmmt25", "olympiadbench", "beyondaime", "apex-shortlist", "hmmt26"):
        assert profiles.EXEMPLAR_SETS[name] == "math"


def test_the_new_tasksets_did_not_perturb_an_existing_rendering() -> None:
    """The digest is per taskset, so adding tasksets cannot move an old one -- but
    that is the property the whole "no re-baseline" claim leans on, so it is
    asserted rather than reasoned about."""

    def digest(taskset: str) -> str:
        return templates.resolve(
            taskset=taskset, template=templates.for_profile("base-kshot", taskset)
        ).sha256()

    assert digest("aime24") == digest("aime26")
    assert digest("math500") == digest("olympiadbench")
    assert digest("gsm8k") != digest("aime26")


def test_the_shared_kshot_stop_strings_are_untouched() -> None:
    """`KSHOT_STOP` is global. Changing it re-baselines every existing number, so
    this suite must be addable without touching it."""
    assert templates.KSHOT_STOP == ("\nProblem:", "\n\nProblem:", "**Problem")


# --- Strict-rung parity, in the shape of tests/test_strict_parity.py ----------

#: Real gold answers from the three added tasksets, paired with completions a
#: model would plausibly emit. AIME 2026 golds are bare integers; HMMT's are bare
#: LaTeX; OlympiadBench's are LaTeX that may carry math-mode delimiters. Each line
#: is a case where our strict rung must agree with the reference verifier that
#: `aime26_v1` actually calls.
REFERENCE_PARITY_VECTORS = [
    # AIME 2026, verbatim golds from the pinned revision.
    ("The answer is \\boxed{277}.", "277"),
    ("\\boxed{62}", "62"),
    ("\\boxed{63}", "62"),
    # HMMT, verbatim golds from the pinned revision.
    ("So \\boxed{\\frac{1}{576}}.", "\\frac{1}{576}"),
    ("\\boxed{-984}", "-984"),
    ("\\boxed{8\\sqrt{10}}", "8\\sqrt{10}"),
    ("\\boxed{2^{25} \\cdot 26!}", "2^{25} \\cdot 26!"),
    ("\\boxed{1-\\frac{2}{\\pi}}", "1-\\frac{2}{\\pi}"),
    ("\\boxed{\\sqrt{\\frac{95}{24}}}", "\\sqrt{\\frac{95}{24}}"),
    # HMMT's one comma-joined pair of roots, which the ladder must not split.
    (
        "\\boxed{\\frac{-1+\\sqrt{17}}{2}, \\frac{-1-\\sqrt{17}}{2}}",
        "\\frac{-1+\\sqrt{17}}{2}, \\frac{-1-\\sqrt{17}}{2}",
    ),
    # OlympiadBench, verbatim golds including the math-mode delimiters it ships.
    ("So the final answer is \\boxed{166}.", "$166$"),
    ("So the final answer is \\boxed{\\frac{41}{12}}.", "$\\frac{41}{12}$"),
    ("So the final answer is \\boxed{2}.", "2"),
    # A wrong answer must stay wrong through the same path.
    ("So the final answer is \\boxed{167}.", "$166$"),
]


@pytest.mark.parametrize(("completion", "gold"), REFERENCE_PARITY_VECTORS)
def test_strict_agrees_with_the_reference_verifier_on_the_new_golds(
    completion: str, gold: str
) -> None:
    """The comparability claim for `math-extended`, on its own data.

    `aime26_v1` scores with `vf.verify_boxed_math_answer`, which is the function
    the strict rung reproduces, so agreement here is what makes an offline AIME
    2026 number mean the same thing as an in-run one. HMMT and OlympiadBench have
    no reference environment at the pin and answer to the same boxed rule by our
    declared choice, so the same vectors pin them too.
    """
    expected = verify_boxed_math_answer(completion, gold) == 1.0
    assert score(completion, gold).strict is expected


def test_hmmt_strict_is_stricter_than_matharenas_published_rule() -> None:
    """The declared divergence, asserted rather than only written down.

    MathArena's grader also accepts `\\fbox` and falls back to a bare trailing
    integer. Ours requires `\\boxed`, so our number is not their leaderboard
    number and must never be quoted against it. If this ever starts passing, the
    divergence has closed and the taskset docstring is wrong.
    """
    assert score("\\fbox{103}", "103").strict is False
    assert score("The answer is 103", "103").strict is False
    assert score("\\boxed{103}", "103").strict is True


# --- The OlympiadBench ladder-loss check -------------------------------------

#: E-034's falsification criterion. The boxed ladder replaces OlympiadBench's own
#: judger, and the question is how much of the benchmark that costs. Exceeding
#: this bound falsifies the claim for this taskset alone.
MAX_LADDER_LOSS = 0.02


def test_the_boxed_ladder_scores_essentially_every_olympiadbench_gold() -> None:
    """Measured over all 674 golds with the repository's own `equivalent()`.

    The completion built here is the one the reference instruction asks for: the
    gold boxed, with any math-mode delimiters dropped, because a model writes
    `\\boxed{166}` where the dataset stores `$166$`. A row fails when a *correct*
    answer would not be scored correct -- which is loss attributable to the
    ladder, not to the model.

    This is the check that can falsify the taskset. If it exceeds the bound the
    answer is to report it, never to raise the bound.
    """
    spec = suites.taskset("math-extended", "olympiadbench")
    try:
        rows = load_raw_rows(spec)
    except Exception as exc:  # noqa: BLE001 - the dataset is simply not present
        pytest.skip(f"OlympiadBench not available at the pinned revision: {exc}")

    assert len(rows) == 674, "the pinned revision moved; the loss bound is not transferable"

    failures = []
    for row in rows:
        gold = render_row(spec, row)[1]
        body = gold.strip()
        if body.startswith("$") and body.endswith("$"):
            body = body[1:-1].strip()
        if not score(f"So the final answer is \\boxed{{{body}}}.", gold).strict:
            failures.append((row["id"], gold))

    loss = len(failures) / len(rows)
    assert loss <= MAX_LADDER_LOSS, (
        f"boxed ladder loses {loss:.1%} of OlympiadBench golds "
        f"({len(failures)}/{len(rows)}), above the {MAX_LADDER_LOSS:.0%} bound: {failures[:10]}"
    )


def test_every_olympiadbench_gold_holds_exactly_one_answer() -> None:
    """The assumption `answer_list_field` rests on. `_answer` refuses a longer
    list rather than taking `[0]`, so a second element would be gold the ladder
    never scores against."""
    spec = suites.taskset("math-extended", "olympiadbench")
    try:
        rows = load_raw_rows(spec)
    except Exception as exc:  # noqa: BLE001 - the dataset is simply not present
        pytest.skip(f"OlympiadBench not available at the pinned revision: {exc}")

    assert {len(row["final_answer"]) for row in rows} == {1}
    # And the 93 that carry several answers do so inside that one string, which is
    # exactly why the instruction has to ask for them.
    multiple = [row for row in rows if row["is_multiple_answer"]]
    assert len(multiple) == 93


def test_the_added_specs_are_frozen_like_the_existing_ones() -> None:
    """A spec is a transcription. Mutating one at runtime would decouple the
    manifest from what actually ran."""
    for spec in suites.resolve("math-extended"):
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.group_size = 1  # type: ignore[misc]
