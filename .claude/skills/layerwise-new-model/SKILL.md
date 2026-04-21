---
name: layerwise-new-model
description: Onboard a new HuggingFace LLM to the vLLM layerwise benchmarking toolkit at /tianhao/debug/layerwise/layerwise_tools/. Produces per-module (attention / MoE / layernorm) GPU time for a target decoder layer under the canonical "prefill eager, decode cuda-graph" rule. Use when the user asks to bench / profile / analyze per-op latency for a new model on vLLM, or wants to compare attention vs MoE times, or is adding a model to the perf database.
---

# Layerwise onboarding — new vLLM model

Goal: produce `self_attn` / `mlp` / layernorm decode times for a target layer of a new HuggingFace model, with zero or minimal code edits.

Toolkit lives at `/tianhao/debug/layerwise/layerwise_tools/`. Canonical run invokes `vllm/run_vllm_bench.py` with auto-configured monkey-patches.

## Step 1 — Environment check

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
python3 -c "import vllm; print(vllm.__version__)"   # expect 0.19.0
which nsys                                           # expect /usr/local/bin/nsys (2025.x)
```

If `LD_LIBRARY_PATH` isn't set, vllm import fails at the CUDA libs. Export it in every shell invocation.

## Step 2 — Reconnaissance on the model

```bash
MODEL=<hf_id>  # e.g. MiniMaxAI/MiniMax-M2.7

python3 <<PY
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
print('registered:', [a for a in archs if any(s in a for s in "${MODEL%%/*}".split())])
PY
```
If the arch you need isn't registered, the model isn't supported in this vLLM version — bail.

```bash
python3 <<PY
from huggingface_hub import hf_hub_download
import json
c = json.load(open(hf_hub_download("$MODEL", "config.json")))
print("arch:", c.get("architectures"))
print("model_type:", c.get("model_type"))
print("top keys:", list(c.keys()))
# if there's a text_config (multimodal), LM params are nested there
if "text_config" in c:
    tc = c["text_config"]
    print("text_config keys:", list(tc.keys()))
    for k in ("num_hidden_layers","first_k_dense_replace","num_experts",
             "n_routed_experts","num_local_experts","num_experts_per_tok",
             "moe_intermediate_size","intermediate_size","hidden_size",
             "layer_types","model_type"):
        if k in tc: print(f"  text_config.{k} =", tc[k])
else:
    for k in ("num_hidden_layers","first_k_dense_replace","num_experts",
             "n_routed_experts","num_local_experts","num_experts_per_tok",
             "moe_intermediate_size","intermediate_size","hidden_size",
             "layer_types"):
        if k in c: print(f"  {k} =", c[k])
PY
```

Note which applies:
- **Text-only** (flat config): override goes top-level, `{"num_hidden_layers": 4}`.
- **Multimodal** (has `text_config`): override nested, `{"text_config.num_hidden_layers": 4}`, and pass `--no-skip-tokenizer-init` at run time (multimodal loader demands a tokenizer instance).
- **Hybrid attention** (has `layer_types` like `["linear_attention", ..., "full_attention"]`): must also override `layer_types` to length `num_hidden_layers`, placing `full_attention` at the target layer.
- **MoE layer placement**: pick the target layer so it is MoE (not dense). Inspect the model's `DecoderLayer.__init__` at `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/<mod>.py` if unclear — it typically uses `first_k_dense_replace`, `decoder_sparse_step`, `num_experts > 0`, or a per-layer `mlp_only_layers` list to gate MoE.

## Step 3 — Smoke test

```bash
cd /tianhao/debug/layerwise          # run from here; run_vllm_bench.py lives at this path

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
export LAYERWISE_TARGET_LAYERS=3

# Flat-config model
python3 run_vllm_bench.py \
    --model $MODEL \
    --config-overrides '{"num_hidden_layers": 4}' \
    --no-async-scheduling --kv-cache-dtype fp8 \
    --compilation-config '{"mode": 0, "cudagraph_mode": "FULL"}' \
    --batch-size 4 --input-len 1 --output-len 3 --max-model-len 32 \
    --num-iters-warmup 0 --num-iters 1 \
    > /tmp/smoke.log 2>&1

# Multimodal model: add --no-skip-tokenizer-init and nest the override:
#    --config-overrides '{"text_config.num_hidden_layers": 4}'
#    --no-skip-tokenizer-init
```

Check the smoke log for:
```
[vllm-layer-skip] installed GPUModelRunner.load_model patch, targets=[3]
INFO ... Resolved architecture: <ArchName>
[vllm-layer-skip] found layers at <some.path>.layers
[vllm-layer-skip] kept=[3], skipped=3 layers (identity forward); total layers=4
[vllm-layer-skip] registered PytHooks early (compile=NONE path, before graph capture)
Avg latency: <small number> seconds
```

If the run exits 0 with these lines, skip to Step 4. Otherwise diagnose via the gotchas in Step 6.

## Step 4 — Full bench with nsys

```bash
export LAYERWISE_TARGET_LAYERS=3
export LAYERWISE_STEP_MILESTONES=1,16,128    # adjust as needed; each milestone = one measured decode step

nsys profile --cuda-graph-trace=node --force-overwrite=true \
    -o /tianhao/debug/layerwise/nsys_newmodel \
  python3 run_vllm_bench.py \
      --model $MODEL \
      --config-overrides '{"num_hidden_layers": 4}' \
      --no-async-scheduling --kv-cache-dtype fp8 \
      --compilation-config '{"mode": 0, "cudagraph_mode": "FULL"}' \
      --batch-size 128 --input-len 1 --output-len 128 \
      --max-model-len 256 --num-iters-warmup 0 --num-iters 1 \
      > /tianhao/debug/layerwise/newmodel_bench.log 2>&1

nsys export --type sqlite --force-overwrite=true \
    /tianhao/debug/layerwise/nsys_newmodel.nsys-rep
```

For a full past_kv sweep (larger runs, ~10 min), push to:
`--batch-size 128 --input-len 1 --output-len 8192 --max-model-len 8256` and
`LAYERWISE_STEP_MILESTONES=1,16,32,64,128,256,512,1024,2048,4096,8192`.

## Step 5 — Parse

**5a.** Discover module naming first (important for the rollup regex):
```bash
python3 /tianhao/debug/layerwise/layerwise_tools/common/parse_nsys_module.py \
    /tianhao/debug/layerwise/nsys_newmodel.sqlite --top 40
```
Look at the top module paths. Most HF models print names like
`<ArchRoot>.model.layers.3.{self_attn|mlp|input_layernorm|post_attention_layernorm}.*`.

**5b.** HF-standard names, per-step attn vs mlp split (covers Mixtral-style MoE naming too):
```bash
python3 /tianhao/debug/layerwise/layerwise_tools/common/parse_nsys_step_sweep.py \
    /tianhao/debug/layerwise/nsys_newmodel.sqlite \
    --rollup 'layers\.(\d+)\.(self_attn|mlp|block_sparse_moe|input_layernorm|post_attention_layernorm)' \
    --layer 3
```

**5c.** For non-standard names, extend the alternation:
- Mixtral / MiniMax-M2 → `block_sparse_moe` (MoE container name; included in default above)
- Qwen3Next / Qwen3.5 hybrid → `linear_attn` (GDN) alongside `self_attn` (full)
- Mamba → `mixer` or `mamba_mixer`
- GDN → `gdn`
- If unsure: run **step 5a** first to inspect actual attribute paths, then craft regex

**5d.** MoE internals at the target layer (experts / shared_experts / gate / topk etc.):
```bash
python3 /tianhao/debug/layerwise/layerwise_tools/common/parse_nsys_step_sweep.py \
    /tianhao/debug/layerwise/nsys_newmodel.sqlite \
    --rollup 'layers\.3\.mlp\.(\w+)'
```

**5e.** One-shot, no step axis (use when milestones weren't set):
```bash
python3 /tianhao/debug/layerwise/layerwise_tools/common/parse_nsys_module.py \
    /tianhao/debug/layerwise/nsys_newmodel.sqlite \
    --rollup 'layers\.3\.mlp\.(\w+)' --top 30
```

## Step 6 — Gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `[vllm-layer-skip] could not locate decoder layers` | New DecoderLayer class name the introspection didn't match | Add class name to `_RETURN_ARITY` dict in `vllm/vllm_layer_skip_patch.py` (2-tuple is HF default; use `2`) |
| `ValidationError: num_attention_heads` / `text_config extracted` | HF transformers doesn't know `model_type`; it fails to parse text_config | Add `model_type` to `rewrites` dict in `vllm/run_vllm_bench.py` (rewrite to a similar arch HF already knows, e.g. `deepseek_v3`) |
| `Can't load image processor for ...` | Multimodal arch demands `preprocessor_config.json` | Already handled — `config_patch` pulls all non-weight files from HF. Check the file actually downloaded. |
| `NotImplementedError: StringFormatVariable` at first forward | torch.dynamo tried to trace PytHooks's `"{}".format(dict)` | Means compile wasn't OFF. Verify `--compilation-config '{"mode": 0, "cudagraph_mode": "FULL"}'` is passed. |
| `Cudagraph mode FULL_AND_PIECEWISE is not compatible ... Overriding to NONE` | Default cudagraph_mode triggered validator | Explicitly pass `"cudagraph_mode": "FULL"` in compilation-config. |
| `RuntimeError: Unsupported data type of kv cache: bfloat16` | DSA models don't support bf16 KV | Use `--kv-cache-dtype fp8`. |
| Auto-downgrade `CUDAGraphMode.FULL → FULL_DECODE_ONLY` | DSA indexer can't capture prefill shapes | Actually desired — this achieves "prefill eager / decode graph" rule. Ignore. |
| Target layer's `mlp` is small, looks like dense MLP | `num_hidden_layers=4` combined with model's MoE gate means layer 3 isn't MoE | Adjust override: e.g. add `{"first_k_dense_replace": 3}` so only L0..L2 are dense; or pick a different target layer with `LAYERWISE_TARGET_LAYERS`. |
| Live MoE number ~10× smaller than AIC `distributed="balanced"` | Dummy weight gate collapses routing to few experts (models with empty-expert skip like DeepseekV2MoE) | Don't trust live MoE when the model skips empty experts. Use AIC `collect_moe_v2.run_moe_torch(..., distributed="balanced")` instead. Qwen3Next-style MoE is safe (monolithic kernel, doesn't skip). |

## Step 6.5 — Simulating TP / EP on a single GPU (no distributed init)

To get "a rank's view" of compute under TP=N, EP=M, or MoE TP=K without
spinning up multi-GPU: **hack the config to a smaller model that matches
rank-0 shape, run as TP=1 on one card**. No `init_process_group` patching,
no `FakeProcessGroup`, no comm machinery. vLLM just builds a legal
smaller model.

| knob being simulated | config override |
|---|---|
| TP=N (heads split) | `num_attention_heads //= N`, `num_key_value_heads //= N` |
| MoE EP=M | total-experts field `//= M` (field name varies by family, see table below) |
| MoE TP=K | `moe_intermediate_size //= K` |

**Total-experts field names vary by model family**:

| Family | total-experts field |
|---|---|
| DeepseekV2 / GLM-5 / Kimi-K2.* | `n_routed_experts` (`n_shared_experts` stays unchanged — shared expert is replicated per rank, not EP'd) |
| Qwen2Moe / MiniMax-M2 | `num_local_experts` (misleading name — it's the *total* in HF config; verify against the HF config.json before overriding) |
| Qwen3Moe / Qwen3.5 | `num_experts` |

**Example — TP=8, EP=8 rank of MiniMax-M2.7**:
```bash
python3 run_vllm_bench.py \
    --model MiniMaxAI/MiniMax-M2.7 \
    --config-overrides '{"num_hidden_layers": 4, "model_type": "qwen2_moe",
        "num_attention_heads": 6,                    
        "num_key_value_heads": 1,                    
        "num_local_experts": 32                      
    }' \
    --no-async-scheduling --kv-cache-dtype fp8 \
    --compilation-config '{"mode": 0, "cudagraph_mode": "FULL"}' \
    --batch-size 128 --input-len 1 --output-len 128 \
    --max-model-len 256 --num-iters-warmup 0 --num-iters 1
```

**Scope — compute only, comm handled separately**:
- This method measures **per-rank local compute**. Cross-rank NCCL
  (`all_reduce` / `all_gather` / EP `all2all`) is intentionally out of
  scope — collected via a separate workstream (NCCL/comm bench) and
  composed with these numbers downstream when building end-to-end wall
  estimates. Don't try to read comm cost out of this.

**Implementation gotchas**:
- Dividends must be clean: `num_heads % tp == 0`, `num_key_value_heads % tp == 0`, `total_experts % ep == 0`, `moe_inter % moe_tp == 0`. vLLM loader raises on fractional shapes.
- `num_key_value_heads` (= `num_attention_heads` for MHA, or smaller for GQA) must also be divisible by tp.
- Shared experts (Kimi's `n_shared_experts=1`, GLM-5's shared block, Qwen3.5's shared_expert) are **replicated per rank, not EP-sharded** — keep them at full size in the override; only the routed experts field gets divided.
- MoE EP on dummy weights amplifies routing collapse (fewer local experts → fewer distinct active ones). If the model's MoE impl skips empty experts (DeepseekV2 family), verify the per-rank number against AIC `collect_moe_v2.run_moe_torch(..., moe_tp_size=..., moe_ep_size=..., distributed="balanced")`.

## Step 7 — Canonical output for perf-DB (example)

For GLM-5-FP8 / Qwen3.5-A10B-FP8 / Kimi-K2.6 / MiniMax already validated:

| Model | self_attn @ bs=128 past=0 | MoE @ bs=128 past=0 | Attn / MoE backend | Notes |
|---|---|---|---|---|
| GLM-5-FP8 DSA | 621 μs | 471 | FLASHMLA_SPARSE / TRITON | DeepseekV2MoE family — dummy gate collapse; use AIC balanced (4810 μs) for perf-DB |
| Qwen3.5-122B-A10B-FP8 | 169 μs | 1295 | FLASH_ATTN / TRITON | Qwen3NextSparseMoeBlock — doesn't skip empty experts, live number reliable |
| Kimi-K2.6 | 289 μs | 1834 | FLASHMLA / Marlin WNA16 | 384 routed + 1 shared, W4A16; dummy-collapse status unverified |
| MiniMax-M2.7 | 256 μs | 221 (block_sparse_moe) | FLASH_ATTN / TRITON Fp8 | Mixtral-style naming; number suspiciously small → likely DeepseekV2-style empty-expert skip + dummy collapse; re-verify via AIC balanced |

Add new rows from this skill's runs. Remember to:
- Document `past_kv=0` (prefill point, mlp = pure MoE regardless of KV)
- Or a longer sweep row showing self_attn growth vs past_kv
- Annotate dummy-collapse risk for DeepseekV2MoE family

## Files touched (summary)

Zero edits for HF-convention text-only and simple multimodal models. Edits only required for:
- Exotic DecoderLayer class name → `vllm/vllm_layer_skip_patch.py::_RETURN_ARITY`
- Exotic `model_type` string → `vllm/run_vllm_bench.py::rewrites`
- Exotic attribute naming (hybrid attn, mamba) → rollup regex at parse time (no code edit)

When stuck on a new model's peculiarity, **always first inspect `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/<modfile>.py`** for the `DecoderLayer` class and what attribute names / config fields drive its MoE/attention branching — that's the ground truth the toolkit must match.
