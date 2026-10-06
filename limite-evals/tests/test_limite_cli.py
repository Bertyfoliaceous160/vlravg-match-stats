from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from limite_evals import cli, profiles, serve, suites, templates


def test_cli_has_no_manual_profile_or_exporter_options() -> None:
    parser = cli._parser()
    args = parser.parse_args(["paradigma-inc/limite-1b-base"])
    assert not hasattr(args, "profile")
    assert not hasattr(args, "stage")
    assert args.seed == 0
    assert parser.parse_args(["paradigma-inc/limite-1b-base", "--seed", "17"]).seed == 17
    for unavailable_option in (
        "--stage",
        "--profile",
        "--convert-dir",
        "--tokenizer-dir",
        "--artifact-chat-template",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["paradigma-inc/limite-1b-base", unavailable_option, "value"]
            )


def test_only_math_extended_is_launchable() -> None:
    assert tuple(suites.SUITES) == ("math-extended",)
    with pytest.raises(ValueError, match="unknown suite"):
        suites.resolve("unknown-suite")


def test_checkpoint_selected_profiles_choose_expected_templates() -> None:
    args = Namespace(chat_template=None)
    spec = suites.taskset("math-extended", "math500")
    assert cli._requested_templates((spec,), profiles.resolve("base-kshot"), args) == {
        "math500": "base-kshot-math"
    }
    assert cli._requested_templates((spec,), profiles.resolve("chat"), args) == {
        "math500": templates.ARTIFACT
    }


def test_vllm_launch_receives_hub_revision_without_remote_code_or_tokenizer_override(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "bin" / "vllm"
    executable.parent.mkdir()
    executable.write_text("")
    command = serve._launch_command(
        "paradigma-inc/limite-1b-violetto",
        engine_venv=tmp_path,
        port=8123,
        max_model_len=131072,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        data_parallel_size=1,
        chat_template_path=None,
        revision="b1f3d572ccacb6919f4d64c321b70ba034ddaef2",
    )
    assert command[:3] == [str(executable), "serve", "paradigma-inc/limite-1b-violetto"]
    assert command[-2:] == ["--revision", "b1f3d572ccacb6919f4d64c321b70ba034ddaef2"]
    assert "--trust-remote-code" not in command
    assert "--tokenizer" not in command
