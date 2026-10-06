"""Mathematical equivalence of two answer strings, via `math-verify`.

Isolated in its own module for one reason: it is the single place where a
`math-verify` version bump can change a reported number. `MATH_VERIFY_VERSION` is
written into every run summary so a bump is visible in the record rather than
silently re-scoring history.

The wrapper matches prime-rl's `math500_v1/verify.py` exactly: both sides are
wrapped in `\\boxed{}` before parsing, the parse and the verify are each bounded
at `TIMEOUT` seconds, and any exception scores 0.0 rather than propagating. That
last rule is not defensive padding -- `math-verify` raises on pathological LaTeX,
and a rollout must not die because one model emitted an unparsable answer.

**A comparison that hits its own bound scores wrong, and that is invisible from
the return value.** `math-verify` raises `TimeoutException`, a `BaseException`,
from a `signal.alarm` handler, catches it inside `parse` and `verify`, writes
`Timeout during comparison` to its own `logging` logger, and returns `False` --
which is the same `False` a genuine mismatch returns. The `except Exception`
below therefore never sees it (and could not catch it if it did), and nothing in
the returned boolean distinguishes "these answers differ" from "the comparator
gave up". In a real run that message reaches stderr through `logging`'s
last-resort handler, unattributed to any rollout:
`res297-preflight-math-extended-003` emitted 31 of them across 1,096 rollouts and
no artifact of that run records one.

`compare` makes it observable **without touching the call**. The alternative --
passing `raise_on_error=True` so the timeout propagates -- would also abort
`verify`'s loop over the parse candidates on the first pair that raises, where it
currently falls through to the next; that is a scoring change, and this module's
whole reason to exist is that scoring changes here are not free. Watching the
logger changes nothing that runs, so strict stays bit-identical to the reference
by construction.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

from math_verify import parse, verify

# Bound so a slow or pathological parse returns False within a fixed time instead
# of stalling the run. Same value as prime-rl's verifier.
TIMEOUT = 5

#: The logger `math_verify.parser` and `math_verify.grader` warn on, named by its
#: package root so both children reach the watcher below.
MATH_VERIFY_LOGGER = "math_verify"

#: The two warnings that mean a bound was hit rather than an answer rejected,
#: verbatim from `math-verify` 0.9.0. Matched on the prefix because the parse one
#: appends the expression it gave up on.
TIMEOUT_MESSAGES = ("Timeout during comparison", "Timeout during parsing")

try:
    MATH_VERIFY_VERSION = version("math-verify")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
    MATH_VERIFY_VERSION = "unknown"


@dataclass(frozen=True)
class Comparison:
    """One equivalence check: what it decided, and whether it was allowed to finish.

    `timed_out` never changes `equal`. A comparison that ran out of time is a
    mismatch here exactly as it was before this field existed, which is the
    conservative reading and the one every existing number was produced under.
    What the field buys is that the reading is now stated rather than assumed.
    """

    equal: bool
    timed_out: bool


def compare(gold: str, prediction: str, *, timeout: int = TIMEOUT) -> Comparison:
    """Whether `prediction` matches `gold`, and whether the comparator timed out.

    Both arguments are bare answer strings (no `\\boxed{}` wrapper); this adds it.
    """
    with _watch_for_timeout() as watcher:
        try:
            equal = bool(
                verify(
                    parse("\\boxed{" + gold + "}", parsing_timeout=timeout),
                    parse("\\boxed{" + prediction + "}", parsing_timeout=timeout),
                    timeout_seconds=timeout,
                )
            )
        except Exception:
            equal = False
    return Comparison(equal=equal, timed_out=watcher.timed_out)


def equivalent(gold: str, prediction: str, *, timeout: int = TIMEOUT) -> bool:
    """Whether `prediction` is mathematically equivalent to `gold`.

    The verdict alone, for a caller with nowhere to record a timeout. `compare`
    is the same check with the timeout still attached.
    """
    return compare(gold, prediction, timeout=timeout).equal


class _TimeoutWatcher(logging.Handler):
    """A handler that records whether `math-verify` gave up, and emits nothing."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.timed_out = False

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage().startswith(TIMEOUT_MESSAGES):
            self.timed_out = True


@contextmanager
def _watch_for_timeout() -> Iterator[_TimeoutWatcher]:
    """Listen to `math-verify`'s logger for the duration of one comparison.

    The level is raised only where it would otherwise swallow the warning before
    any handler sees it, and restored afterwards, so a caller that configured
    `math_verify` gets its configuration back. Nothing is written anywhere: this
    handler observes, and where the message goes remains whatever the ambient
    `logging` configuration says.
    """
    logger = logging.getLogger(MATH_VERIFY_LOGGER)
    watcher = _TimeoutWatcher()
    level = logger.level
    if not logger.isEnabledFor(logging.WARNING):
        logger.setLevel(logging.WARNING)
    logger.addHandler(watcher)
    try:
        yield watcher
    finally:
        logger.removeHandler(watcher)
        logger.setLevel(level)
