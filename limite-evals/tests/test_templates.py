"""The template registry and rendering identity."""

from __future__ import annotations

import hashlib
import json

import pytest

from limite_evals import templates


PINNED = {
    "base-kshot-gsm8k": "92759a67ab35ff5b9558c7ec5fa8525b950b1e91cefa448ec2df0db3f230a3b2",
    "base-kshot-math": "fae4db28aac4c8f394a80cd9fa03990f5ec08fcea12488ba955e270c915c8642",
}


def test_committed_template_bytes_are_pinned() -> None:
    for name, expected in PINNED.items():
        assert hashlib.sha256(templates.load(name).encode()).hexdigest() == expected


def test_registry_exposes_only_committed_base_templates() -> None:
    assert set(templates.names()) == set(PINNED)


def test_base_kshot_defaults_are_taskset_specific() -> None:
    assert templates.default_template("pretrain", "gsm8k") == "base-kshot-gsm8k"
    assert templates.default_template("pretrain", "math500") == "base-kshot-math"
    assert templates.default_stop("base-kshot-math", "math500") == templates.KSHOT_STOP


def test_unknown_template_names_are_refused() -> None:
    for name in ("unknown", "custom", "missing"):
        with pytest.raises(templates.TemplateResolutionError, match="no committed template"):
            templates.load(name)


def test_rendering_hash_covers_bytes_stops_and_prompt_identity() -> None:
    base = templates.Rendering(name="one", template="{{ messages }}", stop=("STOP",))
    renamed = templates.Rendering(name="two", template=base.template, stop=base.stop)
    different_stop = templates.Rendering(name="one", template=base.template, stop=("OTHER",))
    different_prompt = templates.Rendering(
        name="one", template=base.template, stop=base.stop, prompt_identity=json.dumps({"n": 1})
    )
    assert base.sha256() == renamed.sha256()
    assert base.sha256() != different_stop.sha256()
    assert base.sha256() != different_prompt.sha256()
