"""Serving the artifact, and refusing to evaluate the wrong one.

This repository owns the serve step, and now owns the engine with it, though it
still never imports vLLM into this process. The engine runs as a subprocess
inside a *declared* environment, defaulting to this repository's own, which
carries pinned vLLM and the `LimiteForCausalLM` plugin as ordinary dependencies.
`--engine-venv` survives as an override, for a deliberate parity
check against some other environment, rather than as the path every run takes.

The engine version and build are pinned by this repository's lock and wheel URL
in `pyproject.toml`. `_assert_pinned_engine` holds the bound engine to that pin
before a single completion is served. That assertion is fatal -- unlike
`vllm_version` in the fingerprint below, which stays warning-level because it
compares against what a run *declared*, not against what this repository
resolved.

The launch also carries the rendering: `--chat-template` is how a base
checkpoint that has never seen a chat turn is reached through
`/v1/chat/completions` at all (see `templates`). The template is written into the
run directory rather than passed inline, because the file that was actually
served is part of what makes the run reproducible.

A resolved `Rendering` is handed in rather than a `Profile`, and that is the
whole of F-01's repair reaching this module: the digest gated against here now
covers the bytes about to be served, including the bytes a post-trained
checkpoint brings with it, where before it covered a constant that was the same
for every chat run of every checkpoint.

Because this repository assembles the argv itself, it also gets to say what may
never appear in it. `_assert_no_reasoning_parser` is that list, and today it holds
one entry: vLLM's reasoning parser, which would move a completion's think block
out of `message.content` and into `message.reasoning_content` and leave every
think-aware measurement reading text that never had a `<think>` in it. The check
runs on the assembled command rather than inside the builder, so a passthrough
somebody adds later is covered by the thing already written rather than by
remembering to guard it.

The fingerprint gate is the point of the module. After the engine binds, what it
and the artifact actually report is assembled into an *observed* `Fingerprint`
and compared against the declared one. A mismatch in a fatal field -- served
architecture, template SHA, artifact SHA, `max_model_len`, dtype (decision 13) --
raises before the context manager yields, so no evaluation work and no GPU time
follows a number that would be mislabelled. Non-fatal differences are warned
about and handed back on the handle so they reach the manifest.

`_assert_serving_the_checkpoint` runs ahead of all of it, and it is there because
that gate was found to have a hole underneath it. Most of the observed
fingerprint is read from the artifact's own files rather than from the engine --
architecture and dtype out of `config.json`, both SHAs out of what the run
resolved -- so those fields agree with the declaration no matter *which* engine
answered. Only `max_model_len` and `vllm_version` come off the wire. Six cells of
one preflight once bound to a stranger's vLLM, already listening on this port on
the same worker, and were caught by `max_model_len` alone; had that engine served
8192 the run would have scored somebody else's model and labelled every number a
Limite checkpoint. So the id the engine states it is serving, which was previously
read only to address requests with, is now compared against what this run
resolved, and a mismatch is fatal.

That gate is not sufficient on its own, because the identity it checks is the one
thing an adopted engine can legitimately share. `--chat-template` is passed to the
engine at *launch* and the rollouts go to `/chat/completions`, so the rendering
lives on the engine rather than on the client. A cell that adopted a still-bound
engine serving *the same checkpoint under a different profile* would therefore be
served the previous cell's template, pass the identity gate because the checkpoint
genuinely matches, and pass `template_sha256` because that field is computed from
the run's own `Rendering` -- a real completion, rendered wrong, labelled right.
The matrix is the machine for producing exactly that: one checkpoint, several
profiles, sequential cells.

`_assert_nothing_holds_the_port` closes it, from the only place it can be closed:
before the engine is launched, by refusing to proceed when anything already holds
the address. A held port is one this run's engine could not have bound anyway, so
there is nothing to trade away. It interrogates the address rather than any
process state, which is what makes it independent of teardown -- see its
docstring. `DEFAULT_PORT` is zero beside it, meaning "whichever port is free":
the probe makes an undetected wrong number impossible, and a port nobody else
chose makes the collision rare in the first place.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console

from limite_evals.templates import ARTIFACT, Rendering
from limite_evals_core.schema import Fingerprint

_console = Console(force_terminal=True)


def ok(message: str) -> None:
    _console.print(f"[bold green]OK[/bold green] {message}")


def info(message: str) -> None:
    _console.print(f"[cyan]INFO[/cyan] {message}")


def warn(message: str) -> None:
    _console.print(f"[bold yellow]WARN[/bold yellow] {message}")


def error(message: str) -> None:
    _console.print(f"[bold red]ERROR[/bold red] {message}")


#: The engine version this repository pins, and one half of what replaced
#: decision 5's shared-lockfile guarantee. Bump it only together with the `vllm`
#: pin in `pyproject.toml`, and treat the bump as a re-baseline (decision 16).
PINNED_VLLM_VERSION = "0.28.0"


def _default_engine_venv() -> Path:
    """This process's own environment, which is where the pinned vLLM lives.

    It remains overridable for an explicit numerical parity check against a
    separately managed engine environment. Normal reported runs use this
    repository's resolved environment.
    """
    return Path(os.environ.get("LIMITE_EVALS_ENGINE_VENV") or sys.prefix)


#: The environment the engine runs in, and part of the fingerprint.
DEFAULT_ENGINE_VENV = _default_engine_venv()

#: Loopback only: a worker node is shared, and an engine bound to every
#: interface serves whoever finds the port. Loopback is shared too -- every job
#: on the node has the same `127.0.0.1` -- which is why `DEFAULT_PORT` is what it
#: is and why `_assert_nothing_holds_the_port` exists.
HOST = "127.0.0.1"

#: Zero, meaning "whichever port the kernel says is free", resolved per run by
#: `_choose_port`. It was 8000, and a fixed port on shared loopback is how a
#: preflight's six cells came to adopt a sibling trajectory's engine: same
#: account, same node, same number. A free port is not a guarantee -- binding is
#: first-come and nothing can make a port uncollidable by construction, which is
#: `_assert_nothing_holds_the_port`'s job -- it just stops the collision being
#: the default outcome. An explicit `--port` overrides it and is unchanged: the
#: number is in no digest and no gate, so which port a run bound is not a
#: measurement (compare `--data-parallel-size`).
DEFAULT_PORT = 0
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90

#: One device, which is what every run before this flag existed used. Raising it
#: is a throughput decision and never a scoring one -- see `_launch_command`.
DEFAULT_DATA_PARALLEL_SIZE = 1

#: Generous, because a cold start pays for weight loading plus CUDA graph
#: capture. It is a bound rather than a wait: an engine that has not bound by
#: now has failed, and its log says why.
READINESS_TIMEOUT_S = 900.0
POLL_INTERVAL_S = 2.0
POLL_TIMEOUT_S = 2.0
TERMINATE_TIMEOUT_S = 30.0


class ServeError(RuntimeError):
    """The engine could not be brought up, or did not stay up."""


class FingerprintMismatch(ServeError):
    """The engine is serving something other than what the run declared."""


class EngineVersionMismatch(ServeError):
    """The engine is not the vLLM version this repository pins."""


class ServedModelMismatch(ServeError):
    """The engine that answered is serving something other than this run's checkpoint."""


class PortOccupied(ServeError):
    """Something already holds the address this run's engine was about to bind."""


class ReasoningParserRefused(ServeError):
    """A reasoning parser is configured in front of the engine, or already ran.

    Raised from two places, because the invariant has two ends. `serve` refuses a
    launch line carrying the flag, before the process exists; `runner` refuses a
    completion that arrived pre-split, whoever configured the engine it came
    from. Both are fatal, and neither tries to put the completion back together:
    reassembling `reasoning_content` and `content` into the text the grader was
    written for would silently make a parsed run look like an unparsed one, which
    is the comparability failure this instrument keeps refusing.
    """


@dataclass(frozen=True)
class Server:
    """A bound engine, already checked against the run's declared fingerprint."""

    #: OpenAI-compatible root, `/v1` included, ready to hand to a client.
    base_url: str
    #: The id the engine answers to, which is what requests must name.
    model: str
    fingerprint: Fingerprint
    #: Non-fatal fingerprint differences, `{field: (declared, observed)}`. Empty
    #: on a clean match; never contains a fatal field, since those raise.
    mismatches: dict[str, tuple[object, object]]
    log_path: Path


@contextmanager
def serve(
    checkpoint: str | Path,
    *,
    declared: Fingerprint,
    rendering: Rendering,
    taskset: str,
    artifact_sha256: str,
    run_dir: Path,
    engine_venv: Path = DEFAULT_ENGINE_VENV,
    port: int = DEFAULT_PORT,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    data_parallel_size: int = DEFAULT_DATA_PARALLEL_SIZE,
    tensor_parallel_size: int = 1,
    enforce_eager: bool = False,
    readiness_timeout_s: float = READINESS_TIMEOUT_S,
    revision: str | None = None,
    hf_overrides: Mapping[str, Any] | None = None,
    mtp_speculative_tokens: int | None = None,
    max_num_seqs: int | None = None,
    expected_vllm_version: str = PINNED_VLLM_VERSION,
) -> Iterator[Server]:
    """Serve `checkpoint`, gate it against `declared`, and tear the engine down.

    `max_model_len` and the dtype are taken from `declared` rather than passed
    separately: they are launch arguments *and* fatal fingerprint fields, and
    two sources for one number is how a run ends up serving 4096 while
    reporting 8192. `artifact_sha256` is the caller's measurement of what is on
    disk, and is deliberately not read from `declared` -- that pair is the
    comparison the gate exists to make.
    """
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    template_path = _write_chat_template(artifacts, rendering, taskset)
    log_path = artifacts / f"vllm-{taskset}.log"

    port = _choose_port(port)
    root = f"http://{HOST}:{port}"

    command = _launch_command(
        checkpoint,
        revision=revision,
        engine_venv=engine_venv,
        port=port,
        max_model_len=declared.max_model_len,
        dtype=declared.dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        data_parallel_size=data_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
        chat_template_path=template_path,
        hf_overrides=hf_overrides,
        mtp_speculative_tokens=mtp_speculative_tokens,
        max_num_seqs=max_num_seqs,
    )
    # On the assembled argv rather than inside the builder, so it covers whatever
    # produced it -- this module today, an `--engine-arg` passthrough somebody
    # adds later -- and refuses before the process exists rather than after a
    # parsed completion has already been scored.
    _assert_no_reasoning_parser(command)
    # Last thing before the process exists, and the only gate that can catch an
    # engine serving this very checkpoint under somebody else's rendering.
    _assert_nothing_holds_the_port(root, port, checkpoint)

    # The child writes into the log file directly. A pipe would deadlock: vLLM
    # is chatty during weight loading and nothing here drains it until the
    # process exits, which is precisely what we are waiting for.
    with log_path.open("wb") as log:
        log.write(f"$ {' '.join(command)}\n".encode())
        log.flush()
        info(f"starting engine on port {port}")
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=_engine_env(engine_venv),
        )
        try:
            _await_readiness(process, root, log_path, readiness_timeout_s)
            observed, model = _observe(
                root,
                checkpoint,
                revision=revision,
                rendering=rendering,
                artifact_sha256=artifact_sha256,
                engine_venv=engine_venv,
            )
            # First of the three, because it is the question the other two
            # presuppose. They ask what the engine is and what it was given; this
            # asks whether it is *this run's* engine at all, and a wrong answer
            # here makes every later diagnosis point at the wrong thing.
            _assert_serving_the_checkpoint(root, model, checkpoint)
            # Before the fingerprint gate, because it is the stronger claim: the
            # gate asks whether the engine matches this run's declaration, this
            # asks whether it matches the numerics the repository pinned at all.
            _assert_pinned_engine(observed.vllm_version, expected=expected_vllm_version)
            mismatches = _gate(observed, declared)
            ok(f"engine bound on port {port} serving {model!r}")
            yield Server(
                base_url=f"{root}/v1",
                model=model,
                fingerprint=observed,
                mismatches=mismatches,
                log_path=log_path,
            )
        finally:
            _terminate(process)


def _write_chat_template(artifacts: Path, rendering: Rendering, taskset: str) -> Path | None:
    """The template to serve with, on disk, or None to use the model's own.

    Named for the taskset because a k-shot template embeds that taskset's
    exemplars, so one run directory holds several of them and each has to
    survive as the record of what was served.

    The artifact's own template is written out too, even though it is not passed
    to the engine: it is the case whose bytes were previously pinned by nothing
    at all (F-01), and a run directory that records every other rendering but not
    that one would leave the chat runs -- the post-trained ones -- the only ones
    a reader cannot reconstruct.
    """
    # `stem`, because a caller-supplied rendering is named by its path and a
    # filename built from one would be unusable. A registry name has no suffix
    # and survives unchanged.
    path = artifacts / f"chat_template-{Path(rendering.name).stem}-{taskset}.jinja"
    path.write_text(rendering.template)
    return None if rendering.name == ARTIFACT else path


def _launch_command(
    checkpoint: str | Path,
    *,
    engine_venv: Path,
    port: int,
    max_model_len: int,
    dtype: str,
    gpu_memory_utilization: float,
    data_parallel_size: int,
    chat_template_path: Path | None,
    tensor_parallel_size: int = 1,
    enforce_eager: bool = False,
    revision: str | None = None,
    hf_overrides: Mapping[str, Any] | None = None,
    mtp_speculative_tokens: int | None = None,
    max_num_seqs: int | None = None,
) -> list[str]:
    """The engine's own executable, invoked by absolute path.

    Not `vllm` from `PATH`: the environment is a declared input and part of the
    fingerprint, so an operator's shell must not get to choose which engine
    serves. By default that absolute path lands inside this process's own
    environment; under an override it lands inside the one being compared
    against, which is the only way the comparison means anything.

    `hf_overrides` is passed through verbatim and is deliberately not a general
    escape hatch. The only caller that builds one is `rope`, which rescales the
    artifact's own RoPE basis so a model whose native context leaves no
    completion budget can be served at all, and the run that uses it declares
    itself non-comparable in its manifest. Anything reaching the engine this way
    changes the model being served, so a second producer needs the same
    treatment rather than this argument's convenience.
    """
    console_script = engine_venv / "bin" / "vllm"
    python = engine_venv / "bin" / "python"
    if console_script.exists():
        command = [str(console_script), "serve", str(checkpoint)]
    elif python.exists():
        command = [str(python), "-m", "vllm.entrypoints.openai.api_server", "--model", str(checkpoint)]
    else:
        raise ServeError(f"no vLLM entrypoint in {engine_venv}: neither bin/vllm nor bin/python exists")

    command += [
        "--host",
        HOST,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--dtype",
        dtype,
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    # Replicas rather than shards. The checkpoint is small enough to fit on one
    # device many times over, so the scarce resource is KV cache, not parameter
    # memory: at a 65536 context one sequence costs 3 GB of it, and a single
    # device holds a couple of dozen. Data parallel multiplies that ceiling
    # linearly and leaves each replica's arithmetic exactly as it was, which
    # tensor parallel -- resharding the reduction across devices -- does not.
    # Left off the command line entirely at 1 so a single-device run keeps the
    # launch line it already had.
    if data_parallel_size > 1:
        command += ["--data-parallel-size", str(data_parallel_size)]
    if tensor_parallel_size > 1:
        command += ["--tensor-parallel-size", str(tensor_parallel_size)]
    if enforce_eager:
        command.append("--enforce-eager")
    if max_num_seqs is not None:
        command += ["--max-num-seqs", str(max_num_seqs)]
    if mtp_speculative_tokens is not None:
        command += [
            "--speculative-config",
            json.dumps(
                {"method": "qwen3_5_mtp", "num_speculative_tokens": mtp_speculative_tokens},
                sort_keys=True,
                separators=(",", ":"),
            ),
            "--compilation-config",
            json.dumps(
                {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"},
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]
    if hf_overrides:
        # Sorted and separator-normalised so the same override produces the same
        # launch line, which is what makes a run directory's recorded command
        # comparable with the next run's rather than dependent on dict order.
        command += ["--hf-overrides", json.dumps(hf_overrides, sort_keys=True, separators=(",", ":"))]
    if revision is not None:
        command += ["--revision", revision]
    if chat_template_path is not None:
        command += ["--chat-template", str(chat_template_path)]
    return command


#: Launch-line flags that put vLLM's reasoning parser in front of the engine.
#: `--reasoning-parser` is 0.26's spelling and `--enable-reasoning` the older one
#: that still turns up in copied invocations. Underscores and an `=` are folded
#: before the comparison, and any token *naming* a parser -- the JSON value of
#: `--structured-outputs-config`, which is where 0.26 moved the setting -- is
#: refused by the substring test beside them.
REASONING_PARSER_FLAGS = ("reasoning-parser", "enable-reasoning")


def _assert_no_reasoning_parser(command: Sequence[str]) -> None:
    """Refuse an engine that would hand back a completion already taken apart.

    This instrument reads a completion as raw text out of `message.content`,
    reasoning delimiters and all: `ladder.strip_think`, the A--T tracker and the
    protocols are all written against a string that still contains its `<think>`
    block. A reasoning parser breaks that contract at the source. vLLM moves the
    block into `message.reasoning_content` and hands back a `content` with the
    delimiters gone, so `strip_think` finds nothing to strip, the think-aware
    cases go unreachable, and every one of those numbers is still reported --
    which is E42's shape exactly, arrived at from a third direction.

    So the answer is a refusal and not an accommodation. Teaching `runner` to
    concatenate the two fields would produce a run that scores like an unparsed
    one while having been produced differently, and nothing in the fingerprint
    would say so; there is no rendering digest field for "the engine's parser
    was on". Nothing this repository assembles emits these flags and there is no
    passthrough that could carry one, which is what makes this cheap to hold.
    """
    for token in command:
        normalised = token.replace("_", "-").lower()
        named = (
            normalised.startswith("--")
            and normalised.removeprefix("--").partition("=")[0] in REASONING_PARSER_FLAGS
        )
        if not named and "reasoning-parser" not in normalised:
            continue
        message = (
            f"the engine launch line carries {token!r}, which runs vLLM's reasoning parser in "
            "front of the completions this instrument reads. Completions must arrive as raw text "
            "in message.content, reasoning delimiters included; a parser moves the think block "
            "into message.reasoning_content and returns a content with <think> already gone, so "
            "strip_think, the A--T tracker and every think-aware measurement would read a "
            "completion that never had one -- silently, with numbers still reported (E42's "
            "failure by a third route). Reasoning parsers are forbidden with this instrument: "
            "serve the engine without one rather than reassembling what it split."
        )
        error(message)
        raise ReasoningParserRefused(message)


def _engine_env(engine_venv: Path) -> dict[str, str]:
    """This environment, with the engine venv's own `bin` ahead of it on PATH.

    The launcher is invoked by absolute path so that this process's PATH cannot
    decide which vLLM runs. The cost of that choice is that the engine does not
    get its venv's `bin` either -- and vLLM shells out to siblings that live
    exactly there. FlashInfer compiles kernels at startup by running `ninja`,
    which on this cluster is a venv script and not a system binary, so without
    this the engine dies partway through initialisation on a missing tool.

    Prepended rather than appended: the engine's own venv must win over whatever
    the operator happens to have on PATH.
    """
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(filter(None, [str(engine_venv / "bin"), env.get("PATH", "")]))
    return env


def _choose_port(port: int) -> int:
    """The port to bind: the one asked for, or a free one when none was asked for.

    Zero is the caller saying "whichever", which is `DEFAULT_PORT` and therefore
    every run that does not pass `--port`. The kernel is asked for an ephemeral
    port and it is released again immediately, so this is a suggestion rather
    than a reservation -- the engine binds it a moment later, and
    `_assert_nothing_holds_the_port` is what checks that it still can.

    Deliberately **without** `SO_REUSEADDR`, which is the opposite of the choice
    made in `_can_bind` and for the opposite reason. Here the option would let the
    kernel hand back a port with a connection still lingering in `TIME_WAIT` from
    some earlier server, and there is no reason to accept a port with baggage when
    the whole ephemeral range is available. There, the option is what makes the
    answer honest.
    """
    if port:
        return port
    with socket.socket() as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


def _assert_nothing_holds_the_port(root: str, port: int, checkpoint: Path) -> None:
    """Refuse to launch into an address something else already holds.

    Without this, a run whose port is taken does not fail -- it *succeeds against
    the wrong engine*. `_await_readiness` polls `/health` and cannot tell the
    engine this run started from one that was already there, so the occupant is
    adopted, and the run's own engine either never starts or dies unheard in a log
    nobody reads. Every gate downstream then interrogates a stranger's process.

    It is the only gate that closes the worst version of that, which the identity
    gate cannot: an engine serving *this run's own checkpoint* under a different
    profile. The rendering rides on the engine's launch line rather than on the
    request, so adopting it yields real completions of the right model rendered
    with the wrong template, while `template_sha256` -- computed here, from this
    run's `Rendering` -- goes on agreeing with the declaration. Nothing after the
    launch can detect that, so the refusal has to come before it.

    **It is independent of teardown, by construction and on purpose.** `_terminate`
    proves that the launcher was reaped, not that the address was released: a
    worker that escaped the process group, or a socket the kernel has not finished
    with, outlives a teardown that returned perfectly. So this asks the address
    itself, at the moment of use, and consults no record of what this process
    believes it shut down. There is no "we started that one, it must be gone"
    branch to be wrong. That is what makes the sequential cells of a matrix safe
    on a shared `--port`, which is the case the probe exists for.

    No false positives to trade against: the check is whether this run's engine
    could have bound the address, asked in the way the engine will ask it, so
    anything it refuses is a launch that was going to fail or be adopted.
    """
    if _can_bind(port):
        return
    message = (
        f"{root} is already held before this run's engine was started: {_occupant(root, checkpoint)}. "
        f"HOST is {HOST} and every job on a worker node shares it, so this is routinely a sibling "
        "trajectory of the same account, or the previous cell of this matrix whose engine outlived "
        "its teardown. Launching anyway would not fail -- the readiness probe would adopt whatever "
        "is there and the run would produce numbers from it -- so the run stops here. Leave --port "
        "off to be given a free one, or pass a --port nothing else holds."
    )
    error(message)
    raise PortOccupied(message)


def _can_bind(port: int) -> bool:
    """Whether this run's engine could bind `HOST:port`, asked as the engine asks it.

    `SO_REUSEADDR` is set because uvicorn sets it, and the answer has to be the
    engine's answer rather than a stricter one. Without it a port whose previous
    server left connections in `TIME_WAIT` refuses this bind while still accepting
    the engine's -- and a probe that refused the cell after the one that just
    finished, on a `--port` an operator passed deliberately, would be worked
    around within the day and would then be protecting nothing.

    With the option set, the bind fails only when something is genuinely bound
    there, which is the question being asked. `SO_REUSEADDR` does not permit two
    listeners; that is `SO_REUSEPORT`, which is not set here or by uvicorn.
    """
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((HOST, port))
        except OSError:
            return False
    return True


def _occupant(root: str, checkpoint: Path) -> str:
    """Who holds the address, said as precisely as the occupant allows.

    "Something is already there" leaves an operator with a puzzle; naming the
    model it is serving leaves them with a diagnosis, and the two cases are not
    equally alarming. An occupant serving a *different* model is the incident
    already seen, and it would have been caught after the fact by the identity
    gate. An occupant serving *this* checkpoint is the one nothing downstream can
    catch, so it is called out as the more dangerous finding rather than the more
    reassuring one it looks like.
    """
    try:
        model = _model_card(root)["id"]
    except (httpx.HTTPError, ServeError, KeyError, TypeError, ValueError):
        return (
            "whatever is bound there does not answer /v1/models, so it cannot be identified -- it "
            "may be an engine still starting up, or not an engine at all"
        )
    if _served_identity(model) != _served_identity(str(checkpoint)):
        return f"a vLLM already serving {model!r}, which is not the checkpoint this run resolved"
    return (
        f"a vLLM already serving {model!r} -- this run's own checkpoint, which is the dangerous "
        "case and not the harmless one. The rendering is on the engine's launch line, not on the "
        "request, so adopting that engine would serve this checkpoint under whichever template it "
        "was started with while this run recorded its own rendering digest, and no gate after the "
        "launch could tell"
    )


def _await_readiness(process: subprocess.Popen, root: str, log_path: Path, timeout_s: float) -> None:
    """Block until `/health` answers, or say where to look when it never does.

    The exit check is not redundant with the timeout: an engine that dies on a
    bad architecture or an out-of-memory error is dead in seconds, and waiting
    the full budget to report it wastes an allocation.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        if process.poll() is not None:
            raise ServeError(
                f"engine exited with code {process.returncode} before binding {root}; see {log_path}"
            )
        try:
            if httpx.get(f"{root}/health", timeout=POLL_TIMEOUT_S).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(POLL_INTERVAL_S, remaining))
    raise ServeError(f"engine did not bind {root} within {timeout_s:.0f}s; see {log_path}")


def _observe(
    root: str,
    checkpoint: str | Path,
    *,
    rendering: Rendering,
    artifact_sha256: str,
    engine_venv: Path,
    revision: str | None = None,
) -> tuple[Fingerprint, str]:
    """What is actually being served, plus the model id requests must name.

    The architecture and dtype come from the artifact's own `config.json` rather
    than from the engine, which reports neither over the API. That is the
    stronger reading anyway: it catches a checkpoint saved in a different
    precision than the run declared, which the engine would silently cast and
    then serve at numerics nobody recorded.
    """
    config = (
        _artifact_config(checkpoint, revision=revision)
        if revision is not None
        else _artifact_config(checkpoint)
    )
    card = _model_card(root)
    observed = Fingerprint(
        architecture=config["architectures"][0],
        artifact_sha256=artifact_sha256,
        template_sha256=rendering.sha256(),
        # The engine's value wins because it is the one that truncates prompts;
        # the config's is only the ceiling it was allowed to pick from.
        max_model_len=card.get("max_model_len") or config["max_position_embeddings"],
        dtype=_artifact_dtype(checkpoint, config, revision=revision),
        vllm_version=_engine_version(root),
        engine_venv=str(engine_venv),
        # The artifact tokenizer is identified by the immutable Hub revision (or
        # by the local tokenizer digest recorded in the checkpoint contract).
        tokenizer=None,
        eos_token_ids=None if rendering.terminator is None else list(rendering.terminator),
    )
    return observed, card["id"]


def _artifact_config(checkpoint: str | Path, *, revision: str | None = None) -> dict:
    """The artifact's own `config.json`, on disk or on the hub.

    A reference baseline is a hub repo id the engine fetches for itself, so there
    is no local directory to read. Its architecture and dtype still have to reach
    the fingerprint -- the engine reports neither over the API -- so the same file
    is fetched here. The download is cached by `huggingface_hub`, and the import
    is deferred so a local checkpoint never pays for it.
    """
    local = Path(checkpoint).expanduser() / "config.json"
    if local.exists():
        return json.loads(local.read_text())

    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(str(checkpoint), "config.json", revision=revision)).read_text())


def _artifact_dtype(checkpoint: str | Path, config: dict, *, revision: str | None = None) -> str:
    """Read a declared dtype first, then unambiguous Safetensors storage only.

    A checkpoint's config is the explicit contract and always wins. Some reference
    artifacts omit it, however; in that case the Safetensors header is sufficient
    evidence because it lists tensor dtypes before the payload. Malformed, missing,
    mixed, or unsupported headers deliberately remain unknown for the fingerprint
    gate to reject.
    """
    declared = _dtype(config)
    if declared:
        return declared

    local = Path(checkpoint).expanduser() / "model.safetensors"
    if local.is_file():
        return _safetensors_dtype(local)
    if revision is None:
        return ""

    return _hub_safetensors_dtype(str(checkpoint), revision)


def _safetensors_dtype(path: Path) -> str:
    """Return one normalized dtype from a Safetensors header without reading weights."""
    try:
        with path.open("rb") as weights:
            header_size_raw = weights.read(8)
            if len(header_size_raw) != 8:
                return ""
            header_size = int.from_bytes(header_size_raw, "little")
            if header_size > 100 * 1024 * 1024:
                return ""
            header_raw = weights.read(header_size)
            if len(header_raw) != header_size:
                return ""
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return ""

    return _safetensors_header_dtype(header_raw)


def _hub_safetensors_dtype(repo_id: str, revision: str) -> str:
    """Read only an immutable Hub Safetensors header through byte ranges."""
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo_id, "model.safetensors", revision=revision)
    header_size_raw = _http_range(url, 0, 7)
    if len(header_size_raw) != 8:
        return ""
    header_size = int.from_bytes(header_size_raw, "little")
    if header_size > 100 * 1024 * 1024:
        return ""
    return _safetensors_header_dtype(_http_range(url, 8, 7 + header_size))


def _http_range(url: str, start: int, end: int) -> bytes:
    """Fetch exactly one accepted byte range, refusing a server that sends a full artifact."""
    try:
        with httpx.stream(
            "GET",
            url,
            headers={"Range": f"bytes={start}-{end}"},
            follow_redirects=True,
            timeout=POLL_TIMEOUT_S,
        ) as response:
            expected = f"bytes {start}-{end}/"
            if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                return b""
            raw = b"".join(response.iter_bytes())
    except httpx.HTTPError:
        return b""
    return raw if len(raw) == end - start + 1 else b""


def _safetensors_header_dtype(header_raw: bytes) -> str:
    """Normalize a header's sole storage dtype, or keep incomplete evidence unknown."""
    try:
        header = json.loads(header_raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ""
    if not isinstance(header, dict):
        return ""

    dtypes: set[str] = set()
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(tensor, dict) or not isinstance(tensor.get("dtype"), str):
            return ""
        dtypes.add(tensor["dtype"])
    if len(dtypes) != 1:
        return ""
    return {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}.get(dtypes.pop(), "")


def _model_card(root: str) -> dict:
    models = httpx.get(f"{root}/v1/models", timeout=POLL_TIMEOUT_S).json()["data"]
    if not models:
        raise ServeError(f"engine at {root} serves no model")
    return models[0]


def _engine_version(root: str) -> str | None:
    """What the engine says it is, or None when it will not say.

    None used to be the safe answer, because the version was warning-level
    everywhere it was read. It is now also handed to `_assert_pinned_engine`,
    which refuses it: an engine too old or too broken to expose `/version` is
    one nobody can hold to the pin.
    """
    try:
        return httpx.get(f"{root}/version", timeout=POLL_TIMEOUT_S).json()["version"]
    except (httpx.HTTPError, KeyError, ValueError):
        return None


def _assert_serving_the_checkpoint(root: str, model: str, checkpoint: Path) -> None:
    """Refuse an engine that is serving a model other than the one this run resolved.

    The engine states which model it is serving, and until now that statement was
    only *used* -- read off `/v1/models` to address requests with -- never
    checked. Adopting an identity is not verifying it, and on a shared worker the
    two come apart: `HOST` is loopback, every job on a node shares it, and a
    readiness probe cannot tell the engine this run started from one that was
    already there. That is not hypothetical; see the module docstring.

    What counts as a match is *identity*, not spelling, because a rule that
    refuses an equivalent spelling gets worked around and then protects nothing:

    - A local checkpoint is compared as a resolved path. Trailing slashes, `.`
      and `..` segments, a relative spelling and a symlinked `/work` all name one
      directory, and `Path.resolve` says so. Two paths that resolve to the same
      inode are the same artifact and the comparison must not care how each side
      wrote it down.
    - Anything that is not a path on this filesystem is compared as a name, with
      surrounding whitespace and a trailing slash dropped and case folded away. A
      reference baseline is a hub repo id rather than a directory, the engine
      reports the repo id back, and refusing a legitimate baseline over its
      capitalisation would be the cosmetic strictness this rule is trying to
      avoid.
    - The two forms never compare equal to each other, which is the wanted
      answer: a run that resolved a local directory and an engine reporting a
      repo id are not serving the same thing.

    Fatal, and placed with the other refusals that run between binding and the
    yield, so nothing is rolled out and no GPU time is spent on a number that
    would carry this checkpoint's name over another model's completions. It
    records nothing and gates no field of the fingerprint, so it cannot change
    what a run that passes it computes.
    """
    if _served_identity(model) == _served_identity(str(checkpoint)):
        return
    message = (
        f"the engine answering at {root} reports that it is serving {model!r}, but this run "
        f"resolved {str(checkpoint)!r}. The likely cause is that no engine of this run's ever "
        f"bound: HOST is {HOST} and every job on a worker node shares it, so a vLLM somebody else "
        "started on this port answers the readiness probe and gets adopted -- an engine that came "
        "up in seconds rather than minutes is the other tell, and its log will be empty of a "
        "weight load. Whose completions these would be cannot be recovered afterwards, so the run "
        "is refused before a single rollout. Give this run a port nothing else holds with --port, "
        "or wait for this one to be free."
    )
    error(message)
    raise ServedModelMismatch(message)


def _served_identity(name: str) -> str:
    """One model name reduced to what makes it that model rather than another.

    Two branches because there are two kinds of name -- see
    `_assert_serving_the_checkpoint`, which is the only caller and where the
    reasoning lives. A name that exists on this filesystem is a path and is
    resolved; anything else is a hub repo id and is normalised as a string.
    """
    candidate = Path(name).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return name.strip().rstrip("/").casefold()


def _assert_pinned_engine(observed: str | None, *, expected: str = PINNED_VLLM_VERSION) -> None:
    """Refuse an engine that is not the pinned vLLM, before anything is served.

    Fatal, where the fingerprint's `vllm_version` field is a warning, and the
    difference is not an inconsistency. This repository resolves its own engine,
    so this check is what prevents a reported number from being produced by
    unrecorded serving numerics. A warning here would be insufficient.

    It raises after the engine binds rather than before it launches, because the
    engine's own answer is the ground truth and a venv inspected from outside is
    not -- but still before the context manager yields, so no completion, no
    scored sample and no number follows an engine that failed it.
    """
    if observed is None:
        message = (
            "the engine will not report a version, so it cannot be held to the pinned "
            f"vLLM {expected}"
        )
    elif _public_version(observed) != _public_version(expected):
        message = f"engine serves vLLM {observed}, but this run pins {expected}"
    else:
        return
    error(message)
    raise EngineVersionMismatch(message)


def _public_version(version: str) -> str:
    """A version without its local segment.

    The pinned wheel is `0.28.0+cu129` and the endpoint may report either form.
    Which CUDA build is served is pinned by URL in `pyproject.toml`, which is
    where a build belongs; this comparison is about the version.
    """
    return version.partition("+")[0]


def _dtype(config: dict) -> str:
    """The artifact's storage dtype, stripped of the `torch.` prefix some
    exporters write. An unreadable dtype stays empty and fails the gate: a
    guessed one would be recorded as fact."""
    for section in (config, config.get("text_config")):
        if not isinstance(section, dict):
            continue
        value = section.get("torch_dtype") or section.get("dtype")
        if value:
            return str(value).removeprefix("torch.")
    return ""


def _gate(observed: Fingerprint, declared: Fingerprint) -> dict[str, tuple[object, object]]:
    """Raise on a fatal mismatch, warn on the rest, return what to record."""
    mismatches = observed.mismatches(declared)
    fatal = Fingerprint.fatal(mismatches)
    if fatal:
        error(f"fingerprint mismatch: {_describe(fatal)}")
        raise FingerprintMismatch(f"fatal fingerprint mismatch: {_describe(fatal)}")
    for field, (declared_value, observed_value) in sorted(mismatches.items()):
        warn(f"fingerprint differs on {field}: declared {declared_value!r}, serving {observed_value!r}")
    return mismatches


def _describe(mismatches: dict[str, tuple[object, object]]) -> str:
    return "; ".join(
        f"{field} declared {declared!r} but serving {observed!r}"
        for field, (declared, observed) in sorted(mismatches.items())
    )


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    _signal_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=TERMINATE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        _signal_group(process, signal.SIGKILL)
        process.wait(timeout=TERMINATE_TIMEOUT_S)


def _signal_group(process: subprocess.Popen, sig: int) -> None:
    """Signal the whole session, not just the launcher.

    vLLM's engine-core and worker processes are children, and a signal delivered
    only to the parent can leave one of them holding the GPU memory it reserved,
    which makes the next run in a sweep fail to allocate.
    """
    try:
        os.killpg(os.getpgid(process.pid), sig)
    except (ProcessLookupError, PermissionError):
        process.send_signal(sig)
