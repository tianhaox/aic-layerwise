# TensorRT-LLM Layerwise Bench

This directory is intentionally empty of tooling — TRT-LLM ships its own
layerwise bench driver upstream at:

```
frameworks/TensorRT-LLM/examples/layer_wise_benchmarks/
├── run.py                  # main driver (torch path, no TRT engine)
├── config_ctx.yaml         # prefill  (use_cuda_graph: false)
├── config_gen.yaml         # decode   (use_cuda_graph: true)
├── run.sh                  # nsys wrapper
├── correlation.py / parse.py / parse_e2e.py  # own parsers
└── breakdown_template.html / correlation_template.html  # HTML reports
```

Use that directly. The PyTorch path is default; the legacy TensorRT engine
path (`builder.py` / `trtllm-build`) is not touched here.

**Path-A alignment with sglang / vLLM**:
- CTX (prefill) is eager — satisfies "prefill eager" rule.
- GEN (decode) captures the whole `run_pack()` as ONE full cuda graph via
  `torch.cuda.CUDAGraph()` + `g.replay()` — satisfies "decode cuda graph".

**Per-module attribution**: TRT-LLM's bench does NOT auto-instrument every
nn.Module with NVTX (only emits a single outer `b=... s=... past=...`
annotation + CUDA event wall time). To get per-op breakdown comparable to
the sglang/vLLM side, either:
  1. Add `vllm.utils.nvtx_pytorch_hooks.PytHooks` registration on
     `runner.model` before calling `run_pack()`.
  2. Or use kernel-shortName classification (see
     `../common/compare_dsa_kernels.py` style).

This was noted but not implemented in this toolkit.
