"""Persisted identities for answer-scoring re-baselines.

Scoring writes these pins; aggregation and assembly validate them against
stored protocol records without recomputing or relabeling historical verdicts.
"""

from collections.abc import Iterable, Mapping

from limite_evals_core.protocols import (
    OLYMPIADBENCH_EXACT,
    OLYMPIADBENCH_SCORER_REVISION,
)
from limite_evals_core.schema import Pins, ProtocolRecord, ProtocolSample


def answer_scorer_revisions(tasksets: Iterable[str]) -> str:
    """Return comparison pins only for tasksets with a revised answer scorer."""
    if "olympiadbench" in tasksets:
        return f"olympiadbench={OLYMPIADBENCH_SCORER_REVISION}"
    return ""


def validate_answer_scorer_records(
    pins: Pins,
    taskset: str,
    records: Mapping[str, ProtocolSample | ProtocolRecord],
) -> None:
    """Reject a manifest/record mismatch at a stored-verdict publication boundary.

    Historical unpinned OpenBMB records remain readable and aggregatable. A
    MathArena pin requires its actual exact anchor and exact-based relaxations;
    conversely a MathArena record cannot be published with the historical pin.
    This checks provenance, not the mathematical validity of a stored verdict.
    """
    if taskset != "olympiadbench":
        return
    exact = records.get("exact")
    current_anchor = exact is not None and exact.anchor == OLYMPIADBENCH_EXACT.artefact
    expected = answer_scorer_revisions((taskset,))
    if not pins.answer_scorer_revision and not current_anchor:
        return
    if pins.answer_scorer_revision != expected or not current_anchor:
        raise ValueError(
            "olympiadbench answer scorer provenance mismatch: the MathArena pin "
            "requires MathArena verdicts; rescore into a fresh run directory"
        )
    for name in ("lenient", "permissive"):
        record = records.get(name)
        if record is None or record.relaxes != "exact":
            raise ValueError(f"olympiadbench answer scorer provenance mismatch: {name} must relax exact")
