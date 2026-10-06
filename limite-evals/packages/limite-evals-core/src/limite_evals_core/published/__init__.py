"""Where the published graders' bytes live, so `exact` is constructible anywhere.

The artefacts under `vendor/` are third-party sources, and until now they sat in
`tests/graders/vendor` with `exact_grader` taking the directory as an argument.
That made the anchored protocol a thing only the test suite could build: a caller
outside `tests/` had to know a path inside it, and a library module reaching into
a test tree to find its own data is the wrong dependency in the wrong direction.

They live here instead. `VENDOR_ROOT` is the default `execute_published` and
`exact_grader` resolve to, so nothing has to be told where the bytes are; the
explicit-root form survives for the one test that needs to point at a tampered
copy in a temporary directory.

`vendor/` deliberately is **not** a package. Nothing here imports those files --
`execute_published` reads them, checks their digest against the revision they
claim, and executes a verbatim slice. Making them importable would invite a
future `import`, and an import is a thing a linter, a formatter or an IDE will
eventually offer to fix. The `.ruff.toml` in that directory turns every lint rule
off for the same reason: a fixed byte is a broken digest, and the digest is the
whole claim. `vendor/PROVENANCE.md` carries the table of what came from where.
"""

from __future__ import annotations

from pathlib import Path

#: The directory holding the vendored published graders, resolved from this
#: module rather than from a caller's idea of the repository layout.
VENDOR_ROOT = Path(__file__).parent / "vendor"

__all__ = ["VENDOR_ROOT"]
