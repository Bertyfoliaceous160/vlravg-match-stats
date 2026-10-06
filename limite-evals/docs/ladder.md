# Extraction ladder

The ladder extracts a mathematical answer from the answer region. `strict` uses
the required form; `lenient` applies recovery rules that preserve a plausible
final answer; `permissive` additionally accepts an answer anywhere in the full
completion. Every row below is exercised by `limite_evals_core.ladder.score`.

| Shape | Strict | Lenient | Permissive | Recovery, if any |
|---|:---:|:---:|:---:|---|
| Closed final `\boxed{…}` (including commas or last of several boxes) | yes | yes | yes | — |
| Unclosed think block or unterminated/unbalanced box | no | yes | yes | answer-region or unterminated-box recovery |
| Explicit final answer phrase | no | yes | yes | final-line phrase |
| Inline maths or bare number on the final line | no | yes | yes | final-line recovery |
| Right answer only in closed reasoning, earlier box, or non-final phrase | no | no | yes | `answer_anywhere` |
| No numeric candidate | no | no | no | — |

`permissive` is an upper bound, not a headline: it accepts a correct scratch
answer even when the final answer is wrong. `lenient` separates ordinary format
loss from that case. `\fbox` is not a strict boxed answer here, although some
published graders accept it; `exact` and `reference` are separate anchors, not
rungs of this ladder.

The special case `\boxed{}` is currently recovered as an unterminated box. That
can consume following text and re-baselines only lenient/permissive behavior; it
is documented rather than silently changed.

## Boundaries outside the ladder

- The run's truncation policy is applied before scoring. Under `fail`, no recovery
  can rescue a cap-limited completion; under `score`, the completion is graded.
- `format_ok` is not evaluable when a token-capped, unclosed think block leaves
  no answer region. The rollout remains in every correctness and truncation
  denominator.
- Mathematical comparison is time-bounded; a timeout is wrong and recorded as
  `comparison_timeout`.
- `answer_anywhere` keeps at most 64 deduplicated candidates, preferring the last.

GSM8K runs use the boxed path above. See [`protocols.md`](protocols.md).
