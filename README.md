# aic-layerwise

Per-op GPU time attribution for LLM decode, unified across **vLLM**, **sglang**, and **TensorRT-LLM**. Captures identical `{'Module': '...'}` NVTX ranges via per-`nn.Module` hooks in each framework and attributes CUPTI kernel durations back to the originating module under the canonical *prefill eager · decode cuda-graph* execution rule.

```
layerwise_tools/
├── common/                       # framework-agnostic nsys sqlite parsers
│   ├── config_patch.py           # HF config dotted-key override + aux-file download
│   ├── parse_nsys_step_sweep.py  # per-step × per-module rollup (2-step JOIN via
│   │                             # originalGraphNodeId; handles cuda graph replay)
│   ├── parse_nsys_module.py      # per-module rollup, auto-detects graph vs eager
│   └── parse_nsys_wall_vs_gpu.py # wall (host NVTX) vs GPU kernel sum per step/rank
├── vllm/                         # vLLM driver + monkey-patches
├── vllm_plugin/                  # packaged as `vllm.general_plugins` entry point
│                                 # for spawn-mode worker patch injection
├── sglang/                       # sglang driver + monkey-patches
└── trtllm/                       # points to upstream layer_wise_benchmarks
```

See `layerwise_tools/README.md` for canonical run commands, expected numbers on GLM-5-FP8 / Qwen3.5-122B-A10B-FP8 / Kimi-K2.6 / MiniMax-M2.7, and cross-framework caveats.

The `.claude/skills/layerwise-new-model` skill onboards a new HuggingFace LLM to the vLLM path in 7 steps.
