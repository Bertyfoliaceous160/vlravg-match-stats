"""Integrity checks for the retained published-grader bindings."""

from __future__ import annotations

import hashlib

from limite_evals_core.protocols import PublishedGrader, TASKSET_EXACT
from limite_evals_core.published import VENDOR_ROOT


RETAINED_TASKSETS = {
    "math500",
    "gsm8k",
    "aime24",
    "aime25",
    "aime26",
    "hmmt25",
    "hmmt26",
    "apex-shortlist",
    "olympiadbench",
    "beyondaime",
}


def test_exact_registry_contains_only_retained_tasksets() -> None:
    assert set(TASKSET_EXACT) == RETAINED_TASKSETS


def test_every_bound_published_source_matches_its_pinned_digest() -> None:
    checked = set()
    for declared in TASKSET_EXACT.values():
        if not isinstance(declared, PublishedGrader):
            continue
        for source in declared.sources:
            if source.vendored in checked:
                continue
            checked.add(source.vendored)
            payload = (VENDOR_ROOT / source.vendored).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == source.sha256


def test_retained_math_suite_has_bound_and_explicitly_unavailable_exact_columns() -> None:
    bound = {name for name, value in TASKSET_EXACT.items() if isinstance(value, PublishedGrader)}
    assert bound == {
        "math500",
        "aime25",
        "aime26",
        "hmmt25",
        "hmmt26",
        "apex-shortlist",
        "olympiadbench",
    }
    assert {"gsm8k", "aime24", "beyondaime"}.isdisjoint(bound)
