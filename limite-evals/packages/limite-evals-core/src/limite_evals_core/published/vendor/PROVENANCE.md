# Vendored published graders

This directory contains byte-for-byte copies of the published graders needed by
the `math-extended` suite. The filenames encode the upstream repository
and path; `__` represents a path separator. The pinned digests are also declared
in `limite_evals_core.protocols` and are checked before a grader is executed.

| Local file | Repository | Revision | Upstream path | SHA-256 |
|---|---|---|---|---|
| `hendrycks_math__modeling__math_equivalence.py` | `hendrycks/math` | `985bdc1696e88e8643f081a0ff4719da39f2ae2a` | `modeling/math_equivalence.py` | `c4101b1f51a2bb65665194aecd9f761d668c0c248a0350f196c2950e87488527` |
| `hendrycks_math__modeling__dataset__util.py` | `hendrycks/math` | `985bdc1696e88e8643f081a0ff4719da39f2ae2a` | `modeling/dataset/util.py` | `9183a9ea7bce3c126ac33f993d21bf85cffbdb7bab10088216d286699c9e084a` |
| `hendrycks_math__modeling__eval_math_gpt.py` | `hendrycks/math` | `985bdc1696e88e8643f081a0ff4719da39f2ae2a` | `modeling/eval_math_gpt.py` | `e0a70061d1be9d37681dd07f632d19d177beffc1ee17a10148008d1c95877ead` |
| `openai_prm800k__prm800k__grading__grader.py` | `openai/prm800k` | `7ecc794703b2877f63226f2477a49b34f9b25163` | `prm800k/grading/grader.py` | `9e8bbb6f504ee0d8068e1eca031d78174f1c925cfec391996fdf45c21bb016a9` |
| `openai_prm800k__prm800k__grading__math_normalize.py` | `openai/prm800k` | `7ecc794703b2877f63226f2477a49b34f9b25163` | `prm800k/grading/math_normalize.py` | `13998bf8c35bdc8a76868c84e1704a78567e2552557fda69027cc186568d3a4d` |
| `eth_sri_matharena__src__matharena__parse_manual.py` | `eth-sri/matharena` | `a11194deff8c67a232974a383795e8a2776b4c6f` | `src/matharena/parse_manual.py` | `8858daf09f0b8fda2326853874db13fe32d03b506464edcd8f96215db59a3936` |
| `eth_sri_matharena__src__matharena__parser.py` | `eth-sri/matharena` | `a11194deff8c67a232974a383795e8a2776b4c6f` | `src/matharena/parser.py` | `b3eda3f0a16c171c3a7b1c68c55866463e30007558dd7ea0f36afe29a75f98fe` |
| `eth_sri_matharena__src__matharena__utils.py` | `eth-sri/matharena` | `a11194deff8c67a232974a383795e8a2776b4c6f` | `src/matharena/utils.py` | `fa7c16f177f41b30140d597e78d0336bac88b9c95234836280444165cff79eaf` |
| `eth_sri_matharena__src__matharena__grader.py` | `eth-sri/matharena` | `a11194deff8c67a232974a383795e8a2776b4c6f` | `src/matharena/grader.py` | `ea4af7c475416db0eb3c92a8bb322c53537a7952bccad5c029dd384b39435a64` |

The MathArena configuration files are data rather than executable
graders. They are pinned for the same reason: the options passed to the grader
are part of the evaluation contract.

| Local file | Revision | Upstream path | SHA-256 |
|---|---|---|---|
| `eth_sri_matharena__configs__competitions__aime__aime_2025.yaml` | `a11194deff8c67a232974a383795e8a2776b4c6f` | `configs/competitions/aime/aime_2025.yaml` | `1994936c68f87d5d6710e70b804403ae21bcd14f98c512c8972c11c75e067a1f` |
| `eth_sri_matharena__configs__competitions__aime__aime_2026.yaml` | `a11194deff8c67a232974a383795e8a2776b4c6f` | `configs/competitions/aime/aime_2026.yaml` | `473f860cfb3d0da28fe3886639181b09e9264dc1a15331b8c4593199896cb736` |
| `eth_sri_matharena__configs__competitions__apex__shortlist_2025.yaml` | `a11194deff8c67a232974a383795e8a2776b4c6f` | `configs/competitions/apex/shortlist_2025.yaml` | `f6006853c10fabad2d53dd0f4c98d85c6ff8d075da5172caf8ad70b14e77fb3e` |
| `eth_sri_matharena__configs__competitions__hmmt__hmmt_feb_2025.yaml` | `a11194deff8c67a232974a383795e8a2776b4c6f` | `configs/competitions/hmmt/hmmt_feb_2025.yaml` | `abfe6145f6b564675eba5f9510054059b51ca8a3c71900ff0d191f01c72eafec` |
| `eth_sri_matharena__configs__competitions__hmmt__hmmt_feb_2026.yaml` | `a11194deff8c67a232974a383795e8a2776b4c6f` | `configs/competitions/hmmt/hmmt_feb_2026.yaml` | `9ba66e1ec44653c24a3a6b3e0bb2bb06c17c5fce2f0e5e4ce8e997d65a8ac6d7` |
