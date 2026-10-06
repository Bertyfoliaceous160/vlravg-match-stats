"""Served renderings, their stop strings, and their stable identities.

Base templates are committed files resolved by name. Chat checkpoints use the
template carried by their immutable Hugging Face revision. The checkpoint
registry selects the profile, while an explicit template name or file may
override it.

`Rendering.sha256` covers the exact template bytes, stop strings, terminator,
prompt identity, and chat-template arguments that shape a request. A checkpoint
whose profile requires an artifact template must provide one before the engine
starts. Tests pin the committed template digests and verify that their exemplar
files rebuild the same bytes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from limite_evals_core.schema import Stage

TEMPLATE_DIR = Path(__file__).parent / "templates"

#: The template name meaning "serve the artifact's own". Not a file: the bytes
#: come off the checkpoint, and reading them rather than deferring to them is
#: what gives the digest anything to say.
#:
ARTIFACT = "artifact"

#: Where a checkpoint keeps its template. Both layouts are accepted and compared
#: when both are present.
CHAT_TEMPLATE_FILE = "chat_template.jinja"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

#: Where a checkpoint names the id that ends a completion. It is read for the same
#: reason the template's bytes are: it decides where a completion stops, and nothing
#: else in the fingerprint covers it. `resolve.artifact_sha256` is `config.json` plus
#: the top-level weights **by contract**, so this file is outside it -- and that
#: exclusion is deliberate and must stay, because re-emitting provenance beside an
#: artifact may not change the artifact's identity.
GENERATION_CONFIG_FILE = "generation_config.json"

#: Which committed k-shot template a taskset renders with. Transcribed from
#: `profiles.EXEMPLAR_SETS` rather than imported: `profiles` calls this module
#: now, so an import the other way would be an actual cycle rather than a
#: prospective one. The test asserts the two agree.
KSHOT_TEMPLATES = {
    "gsm8k": "base-kshot-gsm8k",
    "math500": "base-kshot-math",
    "aime24": "base-kshot-math",
    "aime25": "base-kshot-math",
    "aime26": "base-kshot-math",
    "hmmt25": "base-kshot-math",
    "olympiadbench": "base-kshot-math",
    "beyondaime": "base-kshot-math",
    "apex-shortlist": "base-kshot-math",
    "hmmt26": "base-kshot-math",
}

#: Which exemplar file each committed k-shot template was built from. Decision 7's
#: `exemplars_sha256` is computed from the files, and `KSHOT_TEMPLATES` above can
#: only say which file a taskset *defaults* to -- so a run that named a different
#: k-shot template would have reported the default's hash beside a rendering that
#: did not use it. This is what closes that, and it changes nothing for a run that
#: names no template: `base-kshot-math` is `math` and `base-kshot-gsm8k` is
#: `gsm8k`, which is exactly what `EXEMPLAR_SETS` already answered for them.
EXEMPLAR_SET_FOR_TEMPLATE = {
    "base-kshot-gsm8k": "gsm8k",
    "base-kshot-math": "math",
}

#: Transcribed from `profiles.KSHOT_STOP`, and not to be edited. It was derived
#: from `preflight-003`'s truncations, and it is inside the digest of every
#: base-kshot number already reported: changing it re-baselines all of them.
KSHOT_STOP = ("\nProblem:", "\n\nProblem:", "**Problem")

#: The stop strings a template serves with when the caller names none.
#:
#: A caller-supplied file gets none: whoever brings their own template owns its
#: stop strings, and guessing on their behalf is how an unrecorded rendering
#: happens.
STOP_DEFAULTS: dict[str, tuple[str, ...]] = {
    "base-kshot-gsm8k": KSHOT_STOP,
    "base-kshot-math": KSHOT_STOP,
    ARTIFACT: (),
}

STOP_OVERRIDES: dict[str, tuple[str, ...]] = {}


class TemplateResolutionError(RuntimeError):
    """The rendering could not be pinned down, and a guess would go unrecorded."""


@dataclass(frozen=True)
class Rendering:
    """A resolved rendering: the exact text to serve, and the stop strings with it."""

    #: A registry name, `ARTIFACT`, or the path of a caller-supplied file.
    name: str
    #: The bytes to serve, decoded. Never None -- the artifact's own template is
    #: read rather than deferred to, which is the whole repair.
    template: str
    stop: tuple[str, ...]
    #: The identity of anything that shaped the prompt but is **not** in the
    #: served template, already serialised so this stays a frozen, hashable
    #: value. Only a taskset drawing category-matched exemplars has one: its
    #: block varies per question and therefore rides in the prompt, so the
    #: template bytes alone would not move when `n_shot` or the few-shot
    #: instruction changed, and two genuinely different renderings would share a
    #: digest. `None` for every other taskset, which keeps the key out of their
    #: payload entirely rather than writing a null into it.
    prompt_identity: str | None = None
    #: The token ids the served artifact ends a completion on, as
    #: `generation_config.json` declares them. The **other half of the stop
    #: strings**: decision 9 put those in this digest because they decide where a
    #: completion ends, and a terminator decides the same thing one level down.
    #: `None` when the artifact names none, or when the rendering was resolved
    #: without an artifact -- absent from the payload rather than null, exactly as
    #: `prompt_identity` is, so a rendering that has nothing to say here keeps the
    #: digest it already had.
    #:
    terminator: tuple[int, ...] | None = None
    #: Keyword arguments passed to the tokenizer's chat-template renderer.
    #: These can select materially different modes (for example thinking), so
    #: they are canonicalised and included in the rendering identity.
    chat_template_kwargs: tuple[tuple[str, bool], ...] = ()

    def sha256(self) -> str:
        """Identity of the rendering: what was served, and where a completion ends.

        The name is deliberately **not** in the payload. A digest over a label
        manufactures a difference between two profiles that render identically
        while failing to separate two artifacts that do not, which is precisely
        how `Profile.template_sha256` failed. Here two different templates
        cannot share a digest, and two names for the same bytes are the same
        rendering and share one.

        **The terminator is in the payload for the reason the stop strings are**, and
        it closes the surviving half of F-01. The template's bytes were pinned and
        its stop strings with them, but the id the model stops on lives in
        `generation_config.json`, which no fingerprint field covers: `template_sha256`
        did not look at it and `resolve.artifact_sha256` excludes it by a contract
        that must not change. So the same weights served with 151645 and with 151643
        produced manifests that compared equal while scoring 0.0% and 81.2% on
        MATH-500 -- the first capped every rollout at the token budget and the
        truncation rule scored all of them wrong. A digest that separates two chat
        templates but not that has a hole in the one place the rendering axis exists
        to be pinned.

        """
        payload: dict[str, object] = {"template": self.template, "stop": list(self.stop)}
        if self.prompt_identity is not None:
            payload["prompt_identity"] = self.prompt_identity
        if self.terminator is not None:
            payload["terminator"] = list(self.terminator)
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def names() -> list[str]:
    """Every committed template, by name."""
    return sorted(path.stem for path in TEMPLATE_DIR.glob("*.jinja"))


def load(name: str) -> str:
    """A committed template's exact text, or a refusal naming what does exist.

    No fallback. A mistyped name that quietly resolved to a default would serve
    one rendering and record another, which is the failure this module exists to
    remove.
    """
    path = TEMPLATE_DIR / f"{name}.jinja"
    if not path.is_file():
        raise TemplateResolutionError(
            f"no committed template named {name!r}; the registry holds {', '.join(names())}. "
            "To serve a template that is not in it, pass the file itself."
        )
    return path.read_text(encoding="utf-8")


def default_template(stage: Stage, taskset: str) -> str:
    """The template `stage` serves with when the caller names none.

    Only `pretrain` has a committed default, and it depends on the taskset,
    because a k-shot template embeds that taskset's exemplars -- vLLM renders a
    template against `messages` and nothing else, so they cannot be passed at
    request time. Every later stage has been trained against a template and
    serves its own.

    The retained pretraining default is always one of the committed k-shot
    templates; later stages serve the artifact rendering.
    """
    if stage == "pretrain":
        if taskset not in KSHOT_TEMPLATES:
            raise TemplateResolutionError(
                f"no k-shot template for taskset {taskset!r}; there are templates for "
                f"{', '.join(sorted(KSHOT_TEMPLATES))}. Name a template explicitly to render it."
            )
        return KSHOT_TEMPLATES[taskset]
    if stage in ("sft", "posttrain", "reference"):
        return ARTIFACT
    raise TemplateResolutionError(f"unknown stage: {stage!r}")


def for_profile(profile: str, taskset: str) -> str:
    """Which registry entry a rendering profile serves this taskset with.

    The checkpoint allowlist selects the profile. Base and Base Soup use a
    committed k-shot template; Violetto and reference models use the immutable
    artifact template.
    """
    if profile == "chat":
        return ARTIFACT
    if profile != "base-kshot":
        raise TemplateResolutionError(f"unknown rendering profile: {profile!r}")

    if taskset not in KSHOT_TEMPLATES:
        raise TemplateResolutionError(
            f"no k-shot template for taskset {taskset!r}; there are templates for "
            f"{', '.join(sorted(KSHOT_TEMPLATES))}. Name a template explicitly to render it."
        )
    return KSHOT_TEMPLATES[taskset]


def default_stop(name: str, taskset: str) -> tuple[str, ...]:
    """The stop strings a rendering serves with when the caller names none.

    A taskset with boundaries of its own replaces the template's defaults, but
    only where the template has some: a rendering that stops on the model's own
    terminator must not acquire a k-shot boundary merely because of its taskset.
    """
    default = STOP_DEFAULTS.get(name, ())
    if default and taskset in STOP_OVERRIDES:
        return STOP_OVERRIDES[taskset]
    return default


def resolve(
    *,
    taskset: str,
    stage: Stage | None = None,
    template: str | Path | None = None,
    stop: Sequence[str] | None = None,
    artifact: str | Path | None = None,
    artifact_revision: str | None = None,
    prompt_identity: str | None = None,
    chat_template_kwargs: Mapping[str, bool] | None = None,
) -> Rendering:
    """The rendering to serve, with its bytes already in hand.

    `template` is a registry name or the path of a file, and a file wins over
    everything -- it is the escape hatch for a rendering nobody has committed
    yet. `stop` overrides the template's defaults on *any* template, which is
    what no profile but `base-kshot` could ever carry. `artifact` is read only
    when the rendering resolves to the artifact's own template, and is required
    then.

    `stage` is a fallback for a caller with no template in hand; the pipeline
    passes `template=for_profile(...)` and never reaches it. One of the two must
    be given, because a rendering nobody named is a rendering nobody recorded.

    """
    if template is None and stage is None:
        raise TemplateResolutionError(
            "name a template or a stage: a rendering resolved from neither would be a "
            "default nobody chose and nobody recorded"
        )
    requested = template if template is not None else default_template(stage, taskset)

    if isinstance(requested, Path) or _looks_like_a_path(requested):
        path = Path(requested).expanduser()
        if not path.is_file():
            raise TemplateResolutionError(
                f"no such chat template file: {path}. It looks like a path rather than a registry "
                f"name, so it is not being looked up in the registry ({', '.join(names())})."
            )
        name, text = str(path), path.read_text(encoding="utf-8")
    elif requested == ARTIFACT:
        name, text = ARTIFACT, artifact_template(artifact, revision=artifact_revision)
    else:
        name, text = requested, load(requested)

    return Rendering(
        name=name,
        template=text,
        stop=tuple(stop) if stop is not None else default_stop(name, taskset),
        prompt_identity=prompt_identity,
        terminator=artifact_terminator(artifact, revision=artifact_revision),
        chat_template_kwargs=tuple(sorted((chat_template_kwargs or {}).items())),
    )


def artifact_terminator(
    artifact: str | Path | None, *, revision: str | None = None
) -> tuple[int, ...] | None:
    """The ids the checkpoint ends a completion on, or None if it names none.

    Unlike `artifact_template` this **never refuses**. A missing or unreadable
    `generation_config.json` is a rendering with nothing to say about its
    terminator, not a broken one: an artifact may legitimately carry none and let
    the engine fall back to the tokenizer's own, and turning that into a refusal
    would make a fingerprint improvement a reason a previously servable checkpoint
    stops serving. What it must not do is guess a default, because a guessed id in
    the digest would assert something the artifact never said.

    Ordered as declared rather than sorted: `[151643, 151645]` and `[151645, 151643]`
    ask vLLM for the same thing, but the order is what the artifact wrote and this
    records the artifact.
    """
    if artifact is None:
        return None

    local = Path(artifact).expanduser()
    config = (
        _local_generation_config(local)
        if local.is_dir()
        else _hub_generation_config(str(artifact), revision=revision)
    )
    if config is None:
        return None

    declared = config.get("eos_token_id")
    if declared is None:
        return None
    ids = declared if isinstance(declared, list) else [declared]
    return tuple(int(one) for one in ids)


def artifact_template(artifact: str | Path | None, *, revision: str | None = None) -> str:
    """The template the checkpoint carries, or a refusal before any GPU time is spent.

    A local directory is read from disk; anything else is taken for a hub repo id
    and fetched, mirroring how `serve._artifact_config` reaches a reference
    baseline's `config.json`. The import is deferred so a local checkpoint never
    pays for it.
    """
    if artifact is None:
        raise TemplateResolutionError(
            "this rendering serves the artifact's own chat template, but no artifact was given"
        )

    local = Path(artifact).expanduser()
    text = _local_template(local) if local.is_dir() else _hub_template(str(artifact), revision=revision)
    if text is None:
        raise TemplateResolutionError(
            f"{artifact} carries no chat template: no {CHAT_TEMPLATE_FILE}, and no chat_template "
            f"in {TOKENIZER_CONFIG_FILE}. Serving it would bind an engine that answers 400 to "
            "every request; name a committed template instead."
        )
    return text


def _looks_like_a_path(name: str) -> bool:
    """Whether a string was meant as a file rather than as a registry name.

    The same rule as `resolve._looks_like_a_path`, for the same reason: a
    mistyped path that fell through to a name lookup would be reported as an
    unknown template, which sends the operator to fix the wrong thing.
    """
    return name.endswith(".jinja") or "/" in name or name.startswith("~")


def _local_template(directory: Path) -> str | None:
    path = directory / CHAT_TEMPLATE_FILE
    standalone = path.read_text(encoding="utf-8") if path.is_file() else None
    config = directory / TOKENIZER_CONFIG_FILE
    configured = (
        _template_field(json.loads(config.read_text(encoding="utf-8")), str(config))
        if config.is_file()
        else None
    )
    return _unambiguous_template(standalone, configured, str(directory))


def _hub_template(repo_id: str, *, revision: str | None) -> str | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    def fetch(filename: str) -> Path | None:
        try:
            return Path(hf_hub_download(repo_id, filename, revision=revision))
        except EntryNotFoundError:
            return None

    template_path = fetch(CHAT_TEMPLATE_FILE)
    config_path = fetch(TOKENIZER_CONFIG_FILE)
    standalone = template_path.read_text(encoding="utf-8") if template_path is not None else None
    configured = (
        _template_field(
            json.loads(config_path.read_text(encoding="utf-8")), f"{repo_id}/{TOKENIZER_CONFIG_FILE}"
        )
        if config_path is not None
        else None
    )
    return _unambiguous_template(standalone, configured, repo_id)


def _local_generation_config(directory: Path) -> dict | None:
    path = directory / GENERATION_CONFIG_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _hub_generation_config(repo_id: str, *, revision: str | None) -> dict | None:
    """A hub artifact's generation config, or None rather than an exception.

    **The catch is deliberately broad, and the breadth is the contract.** Unlike
    `_hub_template`, whose failure means the run has no bytes to serve and must stop,
    this contributes one fingerprint field and nothing else. It now runs on every
    rendering, including base ones against a hub reference baseline -- so a repo that
    carries no generation config, an id that is not a well-formed repo id, an offline
    node or a rate limit would each become a reason a run that could have served does
    not. None of those may be. Every one of them means the same thing here: the
    artifact said nothing about its terminator, so the key stays out of the digest.
    """
    from huggingface_hub import hf_hub_download

    try:
        path = Path(hf_hub_download(repo_id, GENERATION_CONFIG_FILE, revision=revision))
    except Exception:
        return None
    return _local_generation_config(path.parent)


def _template_field(config: dict, where: str) -> str | None:
    """The `chat_template` field, refusing the shapes this module cannot pin.

    Recent transformers allows a list of named templates. Picking one of them
    here would be choosing a rendering on the operator's behalf, so it is a
    refusal that says how to be explicit instead.
    """
    value = config.get("chat_template")
    if value is None or isinstance(value, str):
        return value
    raise TemplateResolutionError(
        f"the chat_template in {where} is a {type(value).__name__} rather than a string, so which "
        "rendering it means is the operator's choice; pass the template file to serve."
    )


def _unambiguous_template(standalone: str | None, configured: str | None, artifact: str) -> str | None:
    """Return one template, refusing two disagreeing artifact declarations."""
    if standalone is not None and configured is not None and standalone != configured:
        raise TemplateResolutionError(
            f"{artifact} declares different chat templates in {CHAT_TEMPLATE_FILE} and "
            f"{TOKENIZER_CONFIG_FILE}; choosing one would make the rendering ambiguous. "
            "Pass one explicit template file to serve."
        )
    return standalone if standalone is not None else configured
