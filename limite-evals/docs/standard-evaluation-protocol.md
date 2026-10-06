# Standard evaluation protocol

The claim-bearing suite is `math-extended`. The immutable checkpoint registry
selects both stage and rendering profile: `pretrain`/`base-kshot` for Limite
Base and Base Soup, `posttrain`/`chat` for Violetto, and `reference`/`chat` for
external models.

Generation uses base seed `0` unless explicitly overridden. Each rollout
derives a distinct deterministic seed from that base, including the members of
an `avg-k` group.

Do not compare runs unless their manifests are compatible. In particular, do
not mix profiles, sampling policies, group sizes, generation budgets, template
hashes, tokenizer vocabularies, stop/terminator settings, scorer versions, or
dataset revisions.

For a new checkpoint:

1. resolve and validate the model artefact before allocating an engine;
2. resolve every taskset rendering and hash its exact bytes;
3. run each requested taskset into a new output directory;
4. persist each completion as it finishes so an interrupted run can resume;
5. aggregate only complete, compatible samples;
6. retain all grader views and diagnostics declared by the taskset.

GPU allocation and scheduling are outside this protocol. Running on one H100,
two H100s, or another compatible accelerator changes throughput, not the
measurement contract, provided the manifest's numerical environment remains
valid and the engine produces the same declared interface.
