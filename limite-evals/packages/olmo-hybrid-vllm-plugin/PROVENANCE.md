# OLMo Hybrid vLLM compatibility patch

This package is local compatibility code, not a copied model implementation.
It is loaded by vLLM's `vllm.general_plugins` entry-point group in the same API
and worker processes as the existing Limite plugin.

## Inspected upstream contract

The patch is intentionally limited to this installed wheel:

| input | value |
|---|---|
| distribution | `vllm 0.28.0+cu129` |
| module version | `vllm.__version__ == "0.28.0"` |
| wheel | `vllm-0.28.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl` |
| model source | `vllm/model_executor/models/olmo_hybrid.py` SHA-256 `bdb1e965eda2f8445b7740a18f67bfda75d0b206a95a6067ef4bc41743f3c84f` |
| generic loader source | `vllm/model_executor/models/utils.py` SHA-256 `a93f50ddf56cac4992dcb7699da48cf70207c8fb0e533beea8818c8f48b1e2d8` |
| fused GDN loader source | `vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py` SHA-256 `4e5b144d109c6f13431e1113ca8cf28132df14e7988dfaad08008cd1e43b18a1` |

In that source, `OlmoHybridModel.hf_to_vllm_mapper` maps the GDN source tensors
`q_conv1d`, `k_conv1d`, and `v_conv1d` to one fused `conv1d` parameter with
integer shard ids `0`, `1`, and `2`. `WeightsMapper.apply` stores the id on the
source tensor, but `AutoWeightsLoader._load_param` invokes the target loader
with only `(parameter, tensor)`. The target GDN loader expects the id as its
third argument to place each tensor in the correct fused slice.

`OlmoHybridModel.load_weights` otherwise remains the native vLLM method. The
patch sends only those three source tensors to the native mapper, calls the
existing target parameter's `weight_loader(parameter, tensor, shard_id)`, and
streams every other tensor unchanged to the original method. It neither
registers an architecture nor alters the model/configuration registry.

## Failure and refresh rules

Registration fails before weight loading if the module/distribution version,
`load_weights(self, weights)` API, or complete native stacked mapping changes.
To move this package to a new vLLM build: inspect that wheel again, update the
table and `EXPECTED_*` contract in `patch.py`, then extend the CPU mapping tests
before running an engine preflight.
