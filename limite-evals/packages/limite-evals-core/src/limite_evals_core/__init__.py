"""Scoring, schema, and statistics for Limite evaluations.

This package is deliberately free of `verifiers`, `torch`, and `vllm`, so
scoring and result inspection do not require the GPU serving environment. The
pipeline that drives a real evaluation lives in the sibling `limite-evals`
distribution.
"""

from limite_evals_core.equivalence import MATH_VERIFY_VERSION, Comparison, compare, equivalent
from limite_evals_core.ladder import (
    AnswerFormat,
    LadderResult,
    extract_boxed,
    format_not_evaluable,
    lenient_answer,
    permissive_candidates,
    score,
    strict_answer,
    strict_answer_hashed,
    strip_think,
    think_unclosed,
)
from limite_evals_core.schema import (
    FATAL_FINGERPRINT_FIELDS,
    LADDER,
    SCORER_VERSION,
    Fingerprint,
    FormatPolicy,
    InstrumentEra,
    InstrumentEraMismatch,
    Interval,
    MetricSet,
    NativeMetric,
    Pins,
    Profile,
    RunManifest,
    RunSummary,
    SampleResult,
    Sampling,
    SpeculativeDecoding,
    Stage,
    TaskSummary,
)
from limite_evals_core.stats import bootstrap_interval, mean_of_groups

__all__ = [
    "FATAL_FINGERPRINT_FIELDS",
    "LADDER",
    "MATH_VERIFY_VERSION",
    "SCORER_VERSION",
    "AnswerFormat",
    "Comparison",
    "Fingerprint",
    "FormatPolicy",
    "InstrumentEra",
    "InstrumentEraMismatch",
    "Interval",
    "LadderResult",
    "MetricSet",
    "NativeMetric",
    "Pins",
    "Profile",
    "RunManifest",
    "RunSummary",
    "SampleResult",
    "Sampling",
    "SpeculativeDecoding",
    "Stage",
    "TaskSummary",
    "bootstrap_interval",
    "compare",
    "equivalent",
    "extract_boxed",
    "format_not_evaluable",
    "lenient_answer",
    "mean_of_groups",
    "permissive_candidates",
    "score",
    "strict_answer",
    "strict_answer_hashed",
    "strip_think",
    "think_unclosed",
]
