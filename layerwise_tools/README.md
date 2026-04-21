# Layerwise Benchmarking Toolkit

Per-op GPU time attribution for LLM decode, unified across three frameworks:
**vLLM**, **sglang**, **TensorRT-LLM**. Each framework has its own driver +
monkey-patches that capture identical `{'Module': '...'}` NVTX ranges via
per-nn.Module hooks; a shared set of parsers consume nsys sqlite output.

## Directory layout

```
layerwise_tools/
├── common/                         # shared parsers (framework-agnostic)
│   ├── config_patch.py             # HF config.json patcher + aux-file download
│   ├── parse_nsys_step_sweep.py    # per-step × per-module rollup (main parser)
│   └── parse_nsys_module.py        # per-module rollup, auto-detects graph vs eager
├── sglang/
│   ├── run_bench_skip.py           # driver wrapping sglang.bench_one_batch
│   ├── layer_skip_patch.py         # identity-skip non-target DecoderLayer
│   └── skip_prefill_patch.py       # fake ModelRunner.forward for EXTEND
├── vllm/
│   ├── run_vllm_bench.py           # driver wrapping vllm.benchmarks.latency
│   ├── vllm_layer_skip_patch.py    # identity-skip + early PytHooks (compile=NONE)
│   └── vllm_step_marker.py         # outer NVTX `bench_step::*` at milestone steps
└── trtllm/
    └── README.md                   # points to upstream examples/layer_wise_benchmarks
```

## Unification rule (all three frameworks)

**prefill eager · decode cuda-graph**

| framework | how it achieves this |
|---|---|
| TRT-LLM (`examples/layer_wise_benchmarks`) | `config_ctx.yaml` sets `use_cuda_graph: false`, `config_gen.yaml` sets `use_cuda_graph: true`. Full graph via `torch.cuda.CUDAGraph()`. |
| sglang (this toolkit, `run_bench_skip.py`) | prefill eager (or faked via `skip_prefill_patch`); decode uses `--cuda-graph-max-bs N` / piecewise cuda graph from sglang. |
| vLLM (this toolkit, `run_vllm_bench.py`) | `--compilation-config '{"mode":0,"cudagraph_mode":"FULL"}'`. vLLM auto-downgrades FULL→`FULL_DECODE_ONLY` for DSA models, giving prefill-eager/decode-graph for free. |

## Canonical runs

### vLLM (GLM-5-FP8, bs=128, past_kv sweep)

```bash
cd layerwise_tools/vllm
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
export LAYERWISE_TARGET_LAYERS=3
export LAYERWISE_STEP_MILESTONES=1,16,32,64,128,256,512,1024,2048,4096,8192

nsys profile --cuda-graph-trace=node --force-overwrite=true -o nsys_sweep \
  python3 run_vllm_bench.py \
      --model zai-org/GLM-5-FP8 \
      --config-overrides '{"num_hidden_layers": 4}' \
      --no-async-scheduling --kv-cache-dtype fp8 \
      --compilation-config '{"mode": 0, "cudagraph_mode": "FULL"}' \
      --batch-size 128 --input-len 1 --output-len 8192 \
      --max-model-len 8256 --num-iters-warmup 0 --num-iters 1

nsys export --type sqlite --force-overwrite=true nsys_sweep.nsys-rep

python3 ../common/parse_nsys_step_sweep.py nsys_sweep.sqlite \
    --rollup 'layers\.(\d+)\.(self_attn|mlp|input_layernorm|post_attention_layernorm)' \
    --layer 3
```

**Why `{"mode":0, "cudagraph_mode":"FULL"}`**: `mode=0` (NONE) disables
torch.compile so dynamo doesn't trip on `"{}".format(marker_dict)` in
PytHooks; `FULL` captures the whole decode forward via `torch.cuda.graph()`
(PyTorch-native, no inductor dependency). The compatibility check at
`vllm/config/vllm.py:910` only rejects `cudagraph_mode` values needing
piecewise compilation (PIECEWISE / FULL_AND_PIECEWISE); FULL alone passes.

### sglang (GLM-5-FP8, bs=128, past_kv=8192)

```bash
cd layerwise_tools/sglang
LAYERWISE_TARGET_LAYERS=3 python3 run_bench_skip.py \
    --model-path zai-org/GLM-5-FP8 \
    --config-overrides '{"num_hidden_layers": 4}' \
    --trust-remote-code --load-format dummy \
    --tp-size 1 --enable-layerwise-nvtx-marker \
    --cuda-graph-max-bs 128 \
    --batch-size 128 --input-len 8192 --output-len 3 \
    --run-name glm5_bs128

nsys export --type sqlite --force-overwrite=true nsys_glm5_bs128.nsys-rep

python3 ../common/parse_nsys_module.py nsys_glm5_bs128.sqlite \
    --rollup 'layers\.(\d+)\.(self_attn|mlp|input_layernorm|post_attention_layernorm)'
```

### TensorRT-LLM

Uses the upstream benchmark in `examples/layer_wise_benchmarks/` — see
`trtllm/README.md` for the path-A pointer.

## Environment variables

| Var | Default | Scope | Purpose |
|---|---|---|---|
| `LAYERWISE_TARGET_LAYERS` | `"3"` | sglang, vLLM | Comma-separated layer indices that keep real compute; others get identity forward. |
| `LAYERWISE_SKIP_ENABLE` | `"1"` | sglang, vLLM | Set `0` to disable layer-skip entirely. |
| `LAYERWISE_SKIP_PREFILL` | `"1"` | sglang | Set `0` to disable prefill short-circuit (sglang only). |
| `LAYERWISE_STEP_MARKER` | `"1"` | vLLM | Set `0` to disable step marker wrapping. |
| `LAYERWISE_STEP_MILESTONES` | `"1,16,32,64,128,256,512,1024,2048,4096,8192"` | vLLM | Execute-model call numbers at which to emit `bench_step::*` outer NVTX. |

## Cross-framework perf-DB caveats

- **MoE with dummy weights**: `DeepseekV2MoE` (GLM-5) skips empty experts; dummy
  gate output collapses routing to few experts → 10× under-estimate. For perf-DB,
  use AIC `collect_moe.run_moe_torch(..., distributed="balanced")`.
- **DSA attention**: kv_cache dtype must be `fp8` (bfloat16 not supported by
  `concat_and_cache_mla` in vLLM). AIC collector `--kv-cache-dtype fp8`.
- **Launch overhead visibility**: cuda-graph numbers are production wall;
  eager (`--enforce-eager`) numbers include per-kernel CPU launch overhead
  (~3× inflation at small bs, ~25% at bs=128).

## Not included

- `vllm_skip_prefill_patch.py` was built and discarded — fighting vLLM v1's
  request-oriented scheduler adds too much shadow-state maintenance. vLLM
  path A uses `isl=1` trivial prefill instead.
