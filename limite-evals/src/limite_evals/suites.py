"""Named suites: which tasksets a run covers, how they are prompted, and how scored.

Every field here is transcribed from the reference taskset in
`research-environments` at `f9c43a74`, except GSM8K's prompt and scorer. GSM8K
deliberately adopts MATH-500's appended boxed instruction and boxed verifier;
its prompt revision records that re-baseline. Fields named as choices are used
only where a reference has nothing to transcribe. Group sizes are pinned by the
evaluation contract: an offline avg@32 against an in-run avg@4 is a
different statistic, and pinning the verifier achieves nothing if the group
size drifts instead.

Two things the reference tasksets do **not** agree on, which the contract's
"one instruction, constant across profiles" rule had to absorb:

* The instruction text differs per taskset, and so does where it goes. MATH-500
  appends; AIME24 and AIME25 prepend a different sentence. Decision 8 fixes the
  instruction across *profiles*, not across tasksets -- reproducing each
  taskset's own wording is what keeps its number comparable with the in-run one.
* GSM8K's dataset gold still arrives after `####`, but `_answer` normalises that
  storage format to the bare numeric gold. The model-facing contract is boxed:
  the same appended instruction and reference verifier as MATH-500.

"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from limite_evals_core.ladder import AnswerFormat
from limite_evals_core.schema import PROTOCOL_METRICS, MetricSet

#: An instruction built from the raw dataset row rather than fixed for the taskset.
RowInstruction = Callable[[Mapping[str, Any]], str]

# Verbatim from math500_v1/taskset.py, appended after the problem.
MATH500_INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."
# Verbatim from aime24_v1 and aime25_v1 taskset.py, prepended before the problem.
AIME_INSTRUCTION = (
    "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}.\n\n"
)
# Semantic identity for the GSM8K cutover. The prompt bytes themselves are the
# existing `MATH500_INSTRUCTION`; the revision names the task-specific choice to
# pair them with boxed scoring and boxed exemplars.
GSM8K_PROMPT_REVISION = "limite-evals:gsm8k-math500-boxed-v1"
# Semantic identity for BeyondAIME's Limite-owned prompt contract. The bytes are
# the existing `MATH500_INSTRUCTION`; the revision records that local choice.
BEYONDAIME_PROMPT_REVISION = "limite-evals:beyondaime-math500-boxed-v1"

GSM8K_BOXED_METRICS = MetricSet(
    name="gsm8k-boxed",
    metrics=("reference", "lenient", "permissive", "format_ok", "truncated"),
    correctness=("reference", "lenient", "permissive"),
    requires_answer_region=PROTOCOL_METRICS.requires_answer_region,
    pass_at_k=PROTOCOL_METRICS.pass_at_k,
    row_label=PROTOCOL_METRICS.row_label,
    note=(
        "GSM8K uses the boxed reference as its headline. The benchmark's published "
        "grader requires the retired `#### N` completion format and is unavailable under "
        "this task contract; lenient and permissive therefore relax the boxed reference. "
        "`format_ok` and `truncated` retain their standard meanings."
    ),
)

BEYONDAIME_METRICS = MetricSet(
    name="beyondaime-boxed",
    metrics=("reference", "exact", "lenient", "permissive", "format_ok", "truncated"),
    correctness=("reference", "exact", "lenient", "permissive"),
    requires_answer_region=PROTOCOL_METRICS.requires_answer_region,
    pass_at_k=(),
    row_label=PROTOCOL_METRICS.row_label,
    note=(
        "BeyondAIME reports the boxed `reference` protocol as avg@32. `exact` is "
        "unavailable because the benchmark publishes no executable grader or complete "
        "extraction/equivalence specification; `lenient` and `permissive` relax the same "
        "math-verify comparator. No pass@k or pass^k statistic is part of this protocol. "
        "`format_ok` and `truncated` retain their standard meanings."
    ),
)

# --- OlympiadBench's per-row instruction --------------------------------------
#
# The only taskset here whose instruction is built rather than quoted, and the
# reason `TasksetSpec.instruction` learned to be a callable at all.
#
# **Provenance, which differs from every other taskset in this file.** There is no
# `olympiadbench_v1` in `research-environments` at `f9c43a74` -- the pinned
# checkout has eight maths environments and this is not one of them. The reference
# transcribed below is therefore OlympiadBench's own published evaluator,
# `inference/code/evaluators/evaluator.py` in `OpenBMB/OlympiadBench` at commit
# `ba5b26a7e2849940b598a9159c1190daa2b9175f`, which is the code the benchmark's
# authors used to *pose* the questions. That is a stronger reference than a
# vendored copy would have been, but it is a *different* one, and a reader
# comparing this taskset against the others should know which artefact it answers
# to.
#
# This file builds prompts and scores nothing. The prompt remains OpenBMB's;
# `protocols.OLYMPIADBENCH_EXACT` now applies the pinned AIME MathArena engine
# under Limite's separately recorded scoring contract. Changing the scorer does
# not change this transcription or its source attribution.
#
# Everything below is transcribed from `Evaluator.make_prompt` and
# `get_answer_type_text` for the one config this taskset pins,
# `OE_TO_maths_en_COMP`: open-ended (`OE`), text-only (`TO`), maths, English. The
# reference derives three booleans from that config name -- `is_theorem_proving`,
# `is_math`, `is_chinese` -- so pinning the config fixes them to False/True/False
# and collapses the reference's four prompt branches to the one reproduced here.

#: Verbatim from `english_answer_type_dict`.
_OLYMPIADBENCH_ANSWER_TYPES = {
    "Numerical": "a numerical value",
    "Expression": "an expression",
    "Equation": "an equation",
    "Interval": "an interval",
}


def _olympiadbench_answer_type(answer_type: str) -> str:
    """One answer type in the reference's words (`get_single_answer_type_text`).

    The substring search and its order are the reference's, not a tidier
    equivalent: `in` rather than `==` is what lets a compound type match, and the
    first match wins.

    The single deliberate departure is the failure mode. The reference calls
    `exit()` on an unrecognised type, which inside this harness would kill the
    interpreter and take every rollout already scored with it. Raising keeps the
    same refusal to guess without that cost.
    """
    if "-" in answer_type:
        answer_type = answer_type[: answer_type.find("-")]
    for known in ("Numerical", "Expression", "Equation", "Interval"):
        if known in answer_type:
            return _OLYMPIADBENCH_ANSWER_TYPES[known]
    raise ValueError(f"unrecognised OlympiadBench answer type: {answer_type!r}")


def _olympiadbench_answer_type_text(answer_type: str, multiple_answer: bool) -> str:
    """The answer-type sentence (`get_answer_type_text`), transcribed.

    This is the branch that matters. `multiple_answer` is what turns "The answer
    of The problem should be ..." into "The problem has multiple answers ...", and
    93 of the 674 rows set it. An instruction pinned to the singular wording would
    ask those rows for one answer while scoring them against gold holding several
    -- a 14% systematic error that reads as a model defect rather than a harness
    bug.

    "The answer of The problem" is the reference's own wording, capital and all.
    Correcting the grammar would change the prompt, and a prompt that differs from
    the reference's is no longer the same measurement.
    """
    if "Need_human_evaluate" in answer_type or "Tuple" in answer_type:
        # The reference's own reasoning: "Tuple" means different things in
        # different problems -- a position, or the values of a series of variables
        # -- so naming it in the prompt would mislead more than it guides.
        return ""
    if not multiple_answer:
        return f"The answer of The problem should be {_olympiadbench_answer_type(answer_type)}. "
    if "," not in answer_type:
        return (
            "The problem has multiple answers, each of them should be "
            f"{_olympiadbench_answer_type(answer_type)}. "
        )
    types = [_olympiadbench_answer_type(one) for one in answer_type.split(",")]
    if len(set(types)) == 1:
        return f"The problem has multiple answers, each of them should be {types[0]}. "
    return (
        "The problem has multiple answers, with the answers in order being "
        f"{', '.join(types)}. "
    )


def olympiadbench_instruction(row: Mapping[str, Any]) -> str:
    """OlympiadBench's prompt for one row, from `Evaluator.make_prompt`.

    Built from three row fields -- `is_multiple_answer`, `unit` and `answer_type`
    -- which between them produce nine distinct strings over the pinned 674 rows,
    against the one string a fixed instruction would have offered. Nine is what
    the data yields, not what the code permits: the branches admit more, and the
    absent combinations are simply ones no pinned row exercises.

    The trailing newline is not decoration. The reference joins prompt to question
    with `make_input`'s `prompt + '\\n' + question_content`, so the separator
    belongs to the instruction here, where `render_problem` concatenates the two
    directly.

    `row["unit"]` is tested for truthiness exactly as the reference tests it. That
    works unchanged because the `datasets` loader yields `None` for the 665 rows
    with no unit and a string for the 9 that have one -- read through pandas the
    same column arrives as a float `nan`, which is truthy, and every row would
    silently gain a unit clause it should not have.
    """
    multiple_answer_text = (
        "\\boxed{multiple answers connected with commas}"
        if row["is_multiple_answer"]
        else "\\boxed{answer}"
    )
    unit_text = ""
    if row["unit"]:
        multiple_answer_text += "(unit)"
        unit_text = ", note that the unit of the answer should not be included in \\boxed{}"
    answer_type_text = _olympiadbench_answer_type_text(
        str(row["answer_type"]), bool(row["is_multiple_answer"])
    )
    return (
        "The following is an open-ended problem from an International Math competition. "
        f"{answer_type_text}Please calculate the answer according to the given requirements "
        "and the information provided. Please use LaTeX format to represent the variables "
        "and formulas used in the solution process and results. Please end your solution "
        f'with "So the final answer is {multiple_answer_text}." and give the result '
        f"explicitly{unit_text}.\n"
    )


@dataclass(frozen=True)
class TasksetSpec:
    """Everything a taskset needs that is transcribed from its reference.

    **`instruction` may be a callable, and that knowingly relaxes decision 8 for
    the tasksets that use one.** Decision 8 says the instruction is constant
    across profiles, which the four `MATH_STANDARD` specs satisfy by holding a
    plain string; nothing about them changes here, and a string spec still
    renders through the same one line it always did. The relaxation is scoped to
    tasksets whose reference itself builds the instruction per row: OlympiadBench
    derives at least six variants from three row fields, and 93 of its 674 rows
    carry gold with several answers that only one of those variants asks for.
    Pinning a single string there would not preserve decision 8's symmetry, it
    would only move a systematic 14% error out of sight and into the number. The
    profile side of the symmetry is untouched either way: a callable reads the
    dataset row, never the profile, so the instruction a row gets is still the
    same under `chat` and under `base-kshot`. Decision 8 itself is not amended --
    amending a numbered decision goes through section 16 of the contract.

    The cost is that a callable instruction cannot be rendered from a
    `(problem, answer)` pair alone; `render_problem` needs the raw row, and says
    so loudly rather than falling back to some default variant.
    """

    name: str
    verifiers_id: str
    dataset: str
    split: str
    #: Empty when the reference taskset pins no revision. GSM8K is the only such
    #: case, and it is a gap in the reference rather than an omission here; a
    #: summary records the empty pin so the weakness stays visible.
    revision: str
    group_size: int
    #: A fixed string, or a callable over the raw dataset row when the reference
    #: taskset builds its instruction per row. See the class docstring.
    instruction: str | RowInstruction
    #: "prepend" or "append", matching where the reference puts the instruction.
    instruction_position: str
    #: Which reference verifier the answer protocols reproduce.
    answer_format: AnswerFormat
    #: Source identity for prompt bytes that intentionally differ from the
    #: repository's earlier rendering. Empty means the taskset has no separate
    #: prompt revision to record in the run manifest.
    prompt_revision: str | None = None
    dataset_config: str | None = None
    dataset_subsets: tuple[str, ...] = ()
    #: Names a column whose gold arrives as a list rather than a bare value, as
    #: OlympiadBench's `final_answer` does. Every row of it holds exactly one
    #: element, and `_answer` refuses a longer list instead of taking `[0]`: a
    #: second element would be gold the ladder never scores against, which is a
    #: measurement error and not a formatting quirk.
    answer_list_field: str | None = None
    #: The dataset's licence, when it constrains what may be done with the numbers
    #: it produces. The contract's "Dataset licences" section admitted a
    #: non-commercial dataset on the condition
    #: that the licence stays visible beside any number rather than living only in
    #: a commit message, so it is a field on the spec and not a note in prose.
    #: `None` means the permissive default the rest of the suite already assumes.
    licence: str | None = None
    #: A one-line caveat that has to travel with every number this taskset
    #: produces. OlympiadBench's contamination is the case it exists for: an
    #: risk the a2 agreement accepted rather than resolved, which makes it a
    #: property of the measurement instead of a footnote to it.
    caveat: str | None = None
    problem_field: str | None = None
    #: What this taskset reports. `None` means the five-rung ladder, which is what
    #: every taskset scored by a verifier reports and why the field is not
    #: repeated on the four standard specs.
    metrics: MetricSet | None = None

    def render_problem(self, problem: str, row: Mapping[str, Any] | None = None) -> str:
        """The problem with its instruction, given the raw row if there is one.

        `row` is optional because a string instruction never needs it, which is
        what keeps every existing call site a one-argument call.
        """
        if isinstance(self.instruction, str):
            instruction = self.instruction
        elif row is None:
            raise ValueError(
                f"{self.name}: its instruction is built per row, so render_problem "
                "needs the raw dataset row and cannot render from the problem alone"
            )
        else:
            instruction = self.instruction(row)
        return (
            instruction + problem
            if self.instruction_position == "prepend"
            else problem + instruction
        )


MATH_STANDARD = (
    TasksetSpec(
        name="aime24",
        verifiers_id="aime24-v1",
        dataset="HuggingFaceH4/aime_2024",
        split="train",
        revision="2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
        group_size=32,
        instruction=AIME_INSTRUCTION,
        instruction_position="prepend",
        answer_format="boxed",
    ),
    TasksetSpec(
        name="aime25",
        verifiers_id="aime25-v1",
        dataset="opencompass/AIME2025",
        split="test",
        revision="a6ad95f611d72cf628a80b58bd0432ef6638f958",
        group_size=32,
        instruction=AIME_INSTRUCTION,
        instruction_position="prepend",
        answer_format="boxed",
        dataset_subsets=("AIME2025-I", "AIME2025-II"),
    ),
    TasksetSpec(
        name="math500",
        verifiers_id="math500-v1",
        dataset="HuggingFaceH4/MATH-500",
        split="test",
        revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        group_size=4,
        instruction=MATH500_INSTRUCTION,
        instruction_position="append",
        answer_format="boxed",
    ),
    TasksetSpec(
        name="gsm8k",
        verifiers_id="gsm8k-v1",
        dataset="openai/gsm8k",
        split="test",
        revision="",
        group_size=1,
        instruction=MATH500_INSTRUCTION,
        instruction_position="append",
        prompt_revision=GSM8K_PROMPT_REVISION,
        answer_format="boxed",
        dataset_config="main",
        metrics=GSM8K_BOXED_METRICS,
    ),
)

#: The six tasksets `math-extended` adds. Kept apart from `MATH_STANDARD` so
#: that suite stays literally the tuple it always was: `math-extended` is built by
#: concatenation below, which is what makes "the existing four are untouched" a
#: property of the code rather than a claim to re-verify by reading.
MATH_EXTENDED_ADDITIONS = (
    TasksetSpec(
        # Pure transcription: every field below is read off `aime26_v1/taskset.py`
        # at `f9c43a74`, none is chosen. Its instruction is the same string the
        # AIME 2024 and 2025 tasksets already use, which a test asserts by
        # comparison rather than by eye, and its reward is
        # `verify_boxed_math_answer` -- the exact function the strict rung
        # reproduces. The reference extracts gold as `str(int(row["answer"]))`
        # where the generic `_answer` does `str(...).strip()`; over the pinned 30
        # rows the two agree exactly, which a test also pins.
        name="aime26",
        verifiers_id="aime26-v1",
        dataset="MathArena/aime_2026",
        split="train",
        revision="10b4e45b7a503075d4da8a0d57916a4f06ce6bd2",
        # Mirrors the existing AIME entries, per decision 11.
        group_size=32,
        instruction=AIME_INSTRUCTION,
        instruction_position="prepend",
        answer_format="boxed",
        licence="cc-by-nc-sa-4.0",
    ),
    TasksetSpec(
        # **Two declared choices, not transcriptions.** There is no `hmmt_v1` in
        # `research-environments` at the pin, so unlike every taskset above this
        # one has no reference environment to copy:
        #
        # 1. The instruction is reused from the AIME tasksets rather than
        #    transcribed from an HMMT reference. It is a faithful quote of
        #    `AIME_INSTRUCTION`, but its provenance is a sibling taskset.
        # 2. **The strict rung here will not equal MathArena's published
        #    leaderboard number.** Their grader also accepts `\fbox` and falls
        #    back to a bare trailing integer where ours requires `\boxed`. Ours is
        #    the stricter rule, so the gap is expected to be small and one-signed,
        #    but the two numbers are not interchangeable and quoting ours against
        #    their leaderboard would be a category error.
        #
        # Gold answers are bare LaTeX (`\frac{1}{576}`, `8\sqrt{10}`), not the
        # integers AIME guarantees, so the ladder's equivalence does the work.
        name="hmmt25",
        verifiers_id="hmmt25-v1",
        dataset="MathArena/hmmt_feb_2025",
        split="train",
        revision="6fdc4277120810ff75aa22d2d5489b91f7a262a1",
        group_size=32,
        instruction=AIME_INSTRUCTION,
        instruction_position="prepend",
        answer_format="boxed",
        # Dataset licences: internal evaluation only, no redistribution, no publication
        # of items. Recorded here so it stays beside the number.
        licence="cc-by-nc-sa-4.0",
    ),
    TasksetSpec(
        # Budget transcribed from the reference; instruction built per row.
        #
        # **Contamination is the live caveat.** OlympiadBench was released in
        # February 2024 from public olympiad archives, so any model trained on a
        # web crawl since then may have seen these problems and their solutions. A
        # high number here is evidence of exposure at least as much as of ability,
        # which is why the caveat is a field rather than a remark: the agreement accepted
        # the risk on condition it travels with the number.
        name="olympiadbench",
        verifiers_id="olympiadbench-v1",
        dataset="Hothan/OlympiadBench",
        split="train",
        revision="91184b52131e7fc9455fef848035173aea8cc01a",
        group_size=4,
        instruction=olympiadbench_instruction,
        instruction_position="prepend",
        answer_format="boxed",
        dataset_config="OE_TO_maths_en_COMP",
        answer_list_field="final_answer",
        caveat=(
            "Contamination risk is high: released February 2024 from public olympiad "
            "archives, so a score here may reflect exposure rather than ability."
        ),
    ),
    TasksetSpec(
        # BeyondAIME publishes the problem/gold corpus but no executable scorer.
        # The prompt, boxed comparator and avg@32 aggregation are therefore the
        # Limite protocol explicitly pinned here.
        name="beyondaime",
        verifiers_id="beyondaime-v1",
        dataset="ByteDance-Seed/BeyondAIME",
        split="test",
        revision="c705198ae1043810b1e1693bd879250b51a7a523",
        group_size=32,
        instruction=MATH500_INSTRUCTION,
        instruction_position="append",
        answer_format="boxed",
        prompt_revision=BEYONDAIME_PROMPT_REVISION,
        metrics=BEYONDAIME_METRICS,
        caveat=(
            "The boxed prompt and math-verify scorer are Limite-owned, not an upstream "
            "official protocol; this score is not a replication of upstream Table 2 and "
            "does not establish contamination resistance or verified per-row difficulty."
        ),
    ),
    TasksetSpec(
        # **One declared choice and one caveat; nothing here is transcribed.**
        # MathArena publishes the shortlist and a grading config for it, but there
        # is no matching environment in `research-environments` at the pin, so this
        # taskset has no reference environment to copy -- the same position
        # `hmmt25` above is in.
        #
        # The instruction is reused from the AIME tasksets, exactly as `hmmt25`
        # reuses it. MathArena's own `shortlist_2025.yaml` ships a different one,
        # "Put your final answer within \boxed{}", and declining to adopt it is
        # deliberate: a MathArena taskset rendered differently from its siblings
        # would not be comparable with them inside one suite. The consequence is
        # that no column here replicates the MathArena leaderboard, `exact`
        # included -- that rung binds their grader, but a grader scores the
        # completions it is given and these come from our prompt.
        #
        # **The overlap is why the caveat is a field.** The shortlist is drawn from
        # other 2025 competitions, and four of its 47 problems are ones
        # `math-extended` already scores under `aime25` and `hmmt25`. The published
        # set is kept whole rather than trimmed: a 43-row variant would be a
        # Limite-local benchmark whose number could no longer be quoted against
        # MathArena's own, and that comparability is worth more than the small
        # double-count. So the double-count travels with the number, the way
        # OlympiadBench's contamination risk does.
        name="apex-shortlist",
        verifiers_id="apex-shortlist-v1",
        dataset="MathArena/apex-shortlist",
        split="train",
        revision="f3efdf224ef665f129ddaae37699f6098c65781b",
        group_size=32,
        instruction=AIME_INSTRUCTION,
        instruction_position="prepend",
        answer_format="boxed",
        licence="cc-by-nc-sa-4.0",
        caveat=(
            "Four of the 47 problems are also scored elsewhere in math-extended: "
            "AIME 2025 P14, AIME 2025 P15, HMMT Feb 2025 C9 and HMMT Feb 2025 C10. "
            "A suite total covering this taskset alongside aime25 or hmmt25 counts "
            "them twice. The prompt is Limite's, not MathArena's, so no column here "
            "replicates their leaderboard."
        ),
    ),
    TasksetSpec(
        # The same declared choice as `hmmt25`, for the same reason: no reference
        # environment exists, so the instruction is a faithful quote of
        # `AIME_INSTRUCTION` whose provenance is a sibling taskset rather than an
        # HMMT reference. MathArena's
        # `hmmt_feb_2026.yaml` ships its own instruction and this does not adopt it,
        # for the reason recorded on `apex-shortlist` above.
        #
        # What differs from `hmmt25` is only the competition: February 2026 rather
        # than February 2025. That date is the reason the taskset is worth having.
        # It is after the training cutoff of every checkpoint currently under
        # evaluation, so unlike the 2024 and 2025 entries a score here is evidence
        # of ability rather than of exposure -- which is a property of *when the
        # competition was held*, not a claim this file can keep true forever, and
        # it decays for any future checkpoint trained past February 2026.
        #
        # Gold answers are bare LaTeX (`-\frac{1}{21}`), not the integers AIME
        # guarantees, so the ladder's equivalence does the work exactly as it does
        # for `hmmt25`.
        name="hmmt26",
        verifiers_id="hmmt26-v1",
        dataset="MathArena/hmmt_feb_2026",
        split="train",
        revision="02fba4f74d8e68e73e66a02d540fd979c05c274c",
        group_size=32,
        instruction=AIME_INSTRUCTION,
        instruction_position="prepend",
        answer_format="boxed",
        # Dataset licences: internal evaluation only, no redistribution, no publication
        # of items. Recorded here so it stays beside the number.
        licence="cc-by-nc-sa-4.0",
    ),
)

#: The original standard taskset tuple remains an internal building block; only
#: the extended composition below is part of the public suite registry.
MATH_EXTENDED = MATH_STANDARD + MATH_EXTENDED_ADDITIONS

SUITES = {
    "math-extended": MATH_EXTENDED,
}


def resolve(name: str) -> tuple[TasksetSpec, ...]:
    if name not in SUITES:
        raise ValueError(f"unknown suite: {name!r} (known: {', '.join(sorted(SUITES))})")
    return SUITES[name]


def taskset(suite: str, name: str) -> TasksetSpec:
    for spec in resolve(suite):
        if spec.name == name:
            return spec
    raise ValueError(f"suite {suite!r} has no taskset {name!r}")
