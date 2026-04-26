# DeepSeek-V4 Layerwise Profiling

This note is the local runbook for DeepSeek-V4-Pro experiments under
`layerwise_tools/sglang`.

The main rule is to define the measurement boundary first:

- Full-layer timing: use `run_bench_skip.py` / `run_dsv4pro_tp8_mock.sh`.
- Step/module breakdown: run with `sglang_step_marker.py` enabled and parse nsys.
- AIC op-level DeepGEMM baseline: use `../comparison/collect_moe_deepgemm.py`.
- DeepEP low-latency communication baseline: use AIC/DeepEP
  `aiconfigurator/collector/deep_collector/test_low_latency.py`.

Layerwise experiments in this directory are full-layer SGLang runs. They should
not introduce a standalone MoE layer path. There is no supported "mock MoE
layer" path here. The AIC collector under `../comparison` is an op-level
DeepGEMM compute baseline, not a replacement for the full-layer mock/EP8 traces
and not a DeepEP communication collector.

## Validated Scope

The measured runs in this document were validated on the local H20 setup on
2026-04-24 and 2026-04-26. The validated DeepEP mode is
`--deepep-mode low_latency`. DeepEP `normal` mode is not covered by the result
tables below.

Commands in the `Validated Results` sections are the reproducible commands and
output files used for the reported numbers. Other commands are templates that
use the same wrappers and flags.

## Common Environment

```bash
cd /tianhao/debug/dsv4/aic-layerwise/layerwise_tools/sglang

export SGLANG_APPLY_CONFIG_BACKUP=none
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_DSV4_FP4_EXPERTS=0
export SGLANG_HACK_OVERRIDE_TOPK_IDS_RANDOM=1
export LAYERWISE_TARGET_LAYERS=0
export LAYERWISE_SKIP_PREFILL=1
export LAYERWISE_FLASHMLA_PAD_HQ=1
```

For AIC-aligned MoE routing:

```bash
export LAYERWISE_MOE_TOPK_HACK_MODE=power_law
export LAYERWISE_MOE_TOPK_POWER_LAW_ALPHA=1.01
export LAYERWISE_MOE_TOPK_HACK_SEED=0
export LAYERWISE_MOE_TOPK_HACK_LOG=1
```

For step-level nsys parsing:

```bash
export LAYERWISE_STEP_MARKER=1
export LAYERWISE_STEP_MARKER_SKIP_RUNS=1
export LAYERWISE_STEP_MILESTONES=2
```

For local DeepEP low-latency communication on this container/node:

```bash
export NVSHMEM_IBGDA_NIC_HANDLER=cpu
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_IB_ROCE_VERSION_NUM=2
export NVSHMEM_IB_ADDR_FAMILY=AF_INET
export NVSHMEM_IB_ADDR_RANGE=172.18.0.0/19
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
export NVSHMEM_HCA_LIST=mlx5_0:1
export NCCL_IB_GID_INDEX=3
```

Default GPU-side IBGDA currently fails on this node with NVSHMEM AH/DCT setup
errors. CPU-assisted IBGDA is required to run the local DeepEP LL collector and
EP8 SGLang path, but it is not enough to explain the long communication kernels
seen in EP8 nsys traces.

## Single-GPU Full-Layer Mock

The wrapper is useful for attention/norm/full-layer smoke tests. By default it
runs TP=1 with a local TP8-like config. It does not enable DeepEP.

```bash
MODE=smoke \
GPU=0 \
TARGET_LAYERS=0 \
NUM_LAYERS=1 \
RUN_NAME=dsv4pro_tp8mock_smoke \
RESULT_FILENAME=/tmp/dsv4pro_tp8mock_smoke_result.jsonl \
./run_dsv4pro_tp8_mock.sh
```

To profile it with nsys and export sqlite:

```bash
MODE=smoke \
OUT=/tmp/dsv4pro_tp8mock_smoke_nsys \
SHOW_KERNELS=1 \
MODULE_REGEX='layers\.0\.(self_attn|mlp|mlp\.experts)' \
./profile_dsv4pro_tp8_mock_nsys.sh
```

For the same small MoE shape used in EP experiments, override the config:

```bash
CONFIG_OVERRIDES='{"num_hidden_layers":1,"num_hash_layers":0,"n_hash_layers":0,"num_attention_heads":16,"o_groups":2,"n_routed_experts":16,"num_experts_per_tok":2,"moe_intermediate_size":1024,"vocab_size":10000,"max_position_embeddings":8192,"rope_scaling.original_max_position_embeddings":512}' \
MODE=smoke \
GPU=0 \
TARGET_LAYERS=0 \
RUN_NAME=dsv4pro_tp8mock_same_shape \
RESULT_FILENAME=/tmp/dsv4pro_tp8mock_same_shape_result.jsonl \
./run_dsv4pro_tp8_mock.sh
```

If you force this full-layer single-GPU path to use `--moe-runner-backend
deep_gemm`, its `mlp.experts` range includes standard-path pre/post processing:
sort, `compute_seg_indptr`, `compute_masked_m`, `fill_gateup_input`, and
`post_reorder`. Keep that boundary explicit when comparing against EP8 DeepEP
or the AIC op-level DeepGEMM baseline.

### Validated Single-GPU Mock Breakdown

Measured case:

```text
trace:  /tmp/dsv4_single_mock_ep8_same_shape_deepgemm_step_nsys.sqlite
result: /tmp/dsv4_single_mock_ep8_same_shape_deepgemm_step_nsys_result.jsonl
shape:  bs=1, input_len=16, output_len=2, layer=0
config: num_hidden_layers=1, n_routed_experts=16, topk=2, moe_intermediate_size=1024
note:   LAYERWISE_FLASHMLA_PAD_HQ=1, so FlashMLA attention timing reflects padded h_q.
```

Reproduce with:

```bash
cd /tianhao/debug/dsv4/aic-layerwise/layerwise_tools/sglang

env CUDA_VISIBLE_DEVICES=0 \
  SGLANG_APPLY_CONFIG_BACKUP=none \
  SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
  SGLANG_DSV4_FP4_EXPERTS=0 \
  SGLANG_HACK_OVERRIDE_TOPK_IDS_RANDOM=1 \
  LAYERWISE_TARGET_LAYERS=0 \
  LAYERWISE_SKIP_PREFILL=1 \
  LAYERWISE_FLASHMLA_PAD_HQ=1 \
  LAYERWISE_STEP_MARKER=1 \
  LAYERWISE_STEP_MARKER_SKIP_RUNS=1 \
  LAYERWISE_STEP_MILESTONES=2 \
  nsys profile --cuda-graph-trace=node --force-overwrite=true \
    -o /tmp/dsv4_single_mock_ep8_same_shape_deepgemm_step_nsys \
  python run_bench_skip.py \
    --model-path deepseek-ai/DeepSeek-V4-Pro \
    --config-overrides '{"num_hidden_layers":1,"num_hash_layers":0,"n_hash_layers":0,"num_attention_heads":16,"o_groups":2,"n_routed_experts":16,"num_experts_per_tok":2,"moe_intermediate_size":1024,"vocab_size":10000,"max_position_embeddings":8192,"rope_scaling.original_max_position_embeddings":512}' \
    --trust-remote-code \
    --load-format dummy \
    --tp-size 1 \
    --moe-runner-backend deep_gemm \
    --enable-layerwise-nvtx-marker \
    --mem-fraction-static 0.80 \
    --max-running-requests 4 \
    --max-total-tokens 8192 \
    --cuda-graph-max-bs 1 \
    --cuda-graph-bs 1 \
    --batch-size 1 \
    --input-len 16 \
    --output-len 2 \
    --run-name dsv4_single_mock_ep8_same_shape_deepgemm_step_nsys \
    --result-filename /tmp/dsv4_single_mock_ep8_same_shape_deepgemm_step_nsys_result.jsonl

nsys export \
  --type sqlite \
  --force-overwrite=true \
  --output /tmp/dsv4_single_mock_ep8_same_shape_deepgemm_step_nsys.sqlite \
  /tmp/dsv4_single_mock_ep8_same_shape_deepgemm_step_nsys.nsys-rep
```

Step-2 module breakdown:

| module | GPU time |
|---|---:|
| `0\|input_layernorm` | 2.9 us |
| `0\|self_attn` | 333.8 us |
| `0\|post_attention_layernorm` | 3.0 us |
| `0\|mlp` | 144.4 us |

DSV4/HCA-related full-process kernel breakdown from the same trace:

| area | kernel/module | GPU time |
|---|---|---:|
| mHC pre | `mhc_pre_gemm_sqrsum_splitk_stage_0_kernel` | 29.95 us |
| mHC pre | `mhc_pre_gemm_sqrsum_splitk_stage_1_kernel` | 10.62 us |
| mHC pre | `mhc_pre_big_fuse_tilelang_kernel` | 25.92 us |
| mHC post | `mhc_post_tilelang_kernel` | 21.06 us |
| compressed attention | `self_attn.compressor` module | 21.47 us |
| FlashMLA | `flash_fwd_splitkv_mla_fp8_sparse_kernel` | 24.80 us |
| FlashMLA | `flash_fwd_mla_combine_kernel` | 11.46 us |

MoE submodule breakdown from the same full-layer mock:

| module | GPU time |
|---|---:|
| `0\|mlp.experts` | 90.0 us |
| `0\|mlp.gate` | 8.1 us |
| `0\|mlp.topk` | 17.9 us |
| `0\|mlp.shared_experts.gate_up_proj` | 16.0 us |
| `0\|mlp.shared_experts.act_fn` | 2.1 us |
| `0\|mlp.shared_experts.down_proj` | 9.0 us |

The mock's `mlp.experts` includes standard-path DeepGEMM preparation:

| group | GPU time |
|---|---:|
| DeepGEMM core kernels | 65.70 us |
| sort / seg-indptr / reorder / post-reorder and other prep | 23.58 us |

## 8-GPU DeepEP Low-Latency

Use DeepEP A2A and leave `--moe-runner-backend` unset. This low-latency path
was validated. In current SGLang FP8 MoE, `--moe-a2a-backend deepep` plus
runner `auto` resolves to DeepGEMM.

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  SGLANG_APPLY_CONFIG_BACKUP=none \
  SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
  SGLANG_DSV4_FP4_EXPERTS=0 \
  SGLANG_HACK_OVERRIDE_TOPK_IDS_RANDOM=1 \
  LAYERWISE_MOE_TOPK_HACK_MODE=power_law \
  LAYERWISE_MOE_TOPK_POWER_LAW_ALPHA=1.01 \
  LAYERWISE_MOE_TOPK_HACK_SEED=0 \
  LAYERWISE_MOE_TOPK_HACK_LOG=1 \
  LAYERWISE_STEP_MARKER=1 \
  LAYERWISE_STEP_MARKER_SKIP_RUNS=1 \
  LAYERWISE_STEP_MILESTONES=2 \
  LAYERWISE_TARGET_LAYERS=0 \
  LAYERWISE_SKIP_PREFILL=1 \
  LAYERWISE_FLASHMLA_PAD_HQ=1 \
  NVSHMEM_IBGDA_NIC_HANDLER=cpu \
  NVSHMEM_IB_GID_INDEX=3 \
  NVSHMEM_IB_ROCE_VERSION_NUM=2 \
  NVSHMEM_IB_ADDR_FAMILY=AF_INET \
  NVSHMEM_IB_ADDR_RANGE=172.18.0.0/19 \
  NVSHMEM_ENABLE_NIC_PE_MAPPING=1 \
  NVSHMEM_HCA_LIST=mlx5_0:1 \
  NCCL_IB_GID_INDEX=3 \
  nsys profile --cuda-graph-trace=node --force-overwrite=true \
    -o /tmp/dsv4_tp1dp8ep8_deepep_ll_nsys \
  python run_bench_skip.py \
    --model-path deepseek-ai/DeepSeek-V4-Pro \
    --config-overrides '{"num_hidden_layers":1,"num_hash_layers":0,"n_hash_layers":0,"n_routed_experts":16,"num_experts_per_tok":2,"moe_intermediate_size":1024,"vocab_size":10000,"max_position_embeddings":8192,"rope_scaling.original_max_position_embeddings":512}' \
    --trust-remote-code \
    --load-format dummy \
    --tp-size 8 \
    --dp-size 8 \
    --enable-dp-attention \
    --ep-size 8 \
    --moe-a2a-backend deepep \
    --deepep-mode low_latency \
    --moe-dense-tp-size 1 \
    --enable-layerwise-nvtx-marker \
    --mem-fraction-static 0.06 \
    --max-running-requests 16 \
    --max-total-tokens 8192 \
    --cuda-graph-max-bs 1 \
    --cuda-graph-bs 1 \
    --batch-size 1 \
    --input-len 16 \
    --output-len 2 \
    --run-name dsv4_tp1dp8ep8_deepep_ll \
    --result-filename /tmp/dsv4_tp1dp8ep8_deepep_ll_result.jsonl
```

Export the nsys report:

```bash
nsys export \
  --type sqlite \
  --force-overwrite=true \
  --output /tmp/dsv4_tp1dp8ep8_deepep_ll_nsys.sqlite \
  /tmp/dsv4_tp1dp8ep8_deepep_ll_nsys.nsys-rep
```

Parse step-level modules with rank max:

```bash
python ../common/parse_nsys_step_sweep.py \
  /tmp/dsv4_tp1dp8ep8_deepep_ll_nsys.sqlite \
  --rollup 'layers\.(\d+)\.(self_attn|mlp|input_layernorm|post_attention_layernorm)' \
  --layer 0 \
  --rank-reduce max
```

Parse MoE submodules:

```bash
python ../common/parse_nsys_step_sweep.py \
  /tmp/dsv4_tp1dp8ep8_deepep_ll_nsys.sqlite \
  --rollup 'layers\.(\d+)\.mlp\.(experts|gate|topk|shared_experts\.(?:gate_up_proj|act_fn|down_proj))' \
  --layer 0 \
  --rank-reduce max
```

Use `--per-rank` instead of `--rank-reduce max` when debugging load imbalance.

### Validated DeepEP Results

The table below uses the same small shape:

```text
bs=1, input_len=16, output_len=2
num_hidden_layers=1, n_routed_experts=16, topk=2, moe_intermediate_size=1024
target layer=0, LAYERWISE_SKIP_PREFILL=1
DeepEP mode=low_latency
```

End-to-end `bench_one_batch` results:

| run | runner flags | result file | prefill latency | median decode latency | total latency |
|---|---|---|---:|---:|---:|
| DeepEP LL smoke | `--moe-a2a-backend deepep --deepep-mode low_latency --moe-runner-backend deep_gemm` | `/tmp/dsv4_tp1dp8ep8_deepep_ll_cpuibgda_shm8g_result.jsonl` | 2.336 ms | 2.374 ms | 4.710 ms |
| DeepEP LL auto-runner smoke | `--moe-a2a-backend deepep --deepep-mode low_latency` | `/tmp/dsv4_tp1dp8ep8_deepep_ll_auto_runner_20260424_160628_result.jsonl` | 2.153 ms | 2.400 ms | 4.553 ms |
| DeepEP LL power-law nsys | `--moe-a2a-backend deepep --deepep-mode low_latency`, power-law topk | `/tmp/dsv4_tp1dp8ep8_powerlaw_noswap_lean_nsys_result.jsonl` | 2.328 ms | 2.997 ms | 5.324 ms |

The auto-runner row is the default to use going forward: it validates that
DeepEP low-latency plus runner `auto` resolves to the DeepGEMM MoE runner.

Step-2 rank-max breakdown for the power-law nsys run:

| module | rank-max GPU time |
|---|---:|
| `0\|input_layernorm` | 3.7 us |
| `0\|self_attn` | 339.6 us |
| `0\|post_attention_layernorm` | 3.2 us |
| `0\|mlp` | 79.6 us |

MoE submodule rank-max breakdown:

| module | rank-max GPU time |
|---|---:|
| `0\|mlp.experts` | 39.2 us |
| `0\|mlp.gate` | 6.2 us |
| `0\|mlp.topk` | 6.3 us |
| `0\|mlp.shared_experts.gate_up_proj` | 15.6 us |
| `0\|mlp.shared_experts.act_fn` | 2.1 us |
| `0\|mlp.shared_experts.down_proj` | 9.4 us |

For the same AIC `collect_moe.py` rank0-local power-law workload, routed
expert compute is close to pure DeepGEMM compute. For this distribution, the
AIC helper swaps the hottest EP rank to rank0, so rank0 is also the max-load
rank.

| path | metric | measured time |
|---|---|---:|
| 8-GPU DeepEP LL | `mlp.experts`, rank-max, step=2 | 39.2 us |
| 8-GPU DeepEP LL | DeepGEMM core kernels only, rank-max | 37.79 us |
| AIC `collect_moe_deepgemm.py` | DeepGEMM core `kernel_time_us_per_iter` | 37.98 us |

## Standard DSV4 Full-Layer Alignment

The 2026-04-26 standard DSV4 run uses the real DSV4 MoE dimensions rather than
the earlier reduced smoke config:

```text
experiment: /tianhao/debug/dsv4/experiments_20260426/exp8_dsv4_standard_bs4_isl512_mock_ep8
model:      sgl-project/DeepSeek-V4-Pro-FP8
layer:      num_hidden_layers=1, layer=0
MoE:        hidden=7168, moe_intermediate_size=3072, num_experts=384, topk=6
request:    batch_size=4, input_len=512, output_len=2
analyzed:   decode step=2, past=512
runtime:    max_total_tokens=65536, cuda_graph_bs=4
routing:    AIC power-law alpha=1.01, global MoE tokens=32
```

End-to-end `bench_one_batch`:

| run | prefill latency | median decode latency | total latency |
|---|---:|---:|---:|
| mock full layer | 1.473 ms | 2.853 ms | 4.325 ms |
| EP8 full layer | 2.830 ms | 4.213 ms | 7.043 ms |

Step-2 layer kernel attribution:

| metric | mock | EP8 rank-max | EP8 rank | note |
|---|---:|---:|---:|---|
| layer compute kernels | 1425.277 us | 1452.285 us | 0 | excludes comm kernels |
| layer communication kernels | 0.000 us | 684.638 us | 6 | nsys module-attributed comm residency |
| layer all kernels | 1425.277 us | 1765.050 us | 6 | rank-max with comm counted |

Key module checks:

| module | mock | EP8 compute rank-max | AIC baseline | note |
|---|---:|---:|---:|---|
| `self_attn` | 188.736 us | 191.392 us | - | attention TP/DP shape aligned |
| `mlp.experts` compute | 940.798 us | 941.983 us | 940.747 us | routed expert compute aligned |
| DeepGEMM core `sm90_fp8_gemm_1d2d_impl` | 931.582 us | 921.184 us | - | EP8 rank-max kernel sum |

This validates the compute side of the replacement goal: full-layer mock and
EP8 both match the AIC DeepGEMM op baseline for routed experts. The remaining
difference is how to account for DeepEP communication/wait kernels.

Files:

- `/tianhao/debug/dsv4/experiments_20260426/exp8_dsv4_standard_bs4_isl512_mock_ep8/SUMMARY.md`
- `/tianhao/debug/dsv4/experiments_20260426/exp8_dsv4_standard_bs4_isl512_mock_ep8/summary.csv`
- `/tianhao/debug/dsv4/experiments_20260426/exp8_dsv4_standard_bs4_isl512_mock_ep8/parsed_step2/kernel_compare.csv`

## DeepEP Communication Calibration

The AIC H200 perf database for DeepEP LL is:

```text
/tianhao/debug/dsv4/aiconfigurator/src/aiconfigurator/systems/data/h200_sxm/sglang/0.5.6.post2/wideep_deepep_ll_perf.txt
```

Its exact single-node rows use `hidden=7168`, `topk=8`, `num_experts=256`.
There is no exact H200 DB row for DSV4 Pro's `topk=6`, `num_experts=384`.
`topk` is part of the communication shape because DeepEP dispatch/combine moves
routed token-expert selections; it is not a MoE compute proxy.

The exact local DSV4 communication shape was measured with the original
AIC/DeepEP low-latency collector:

```bash
cd /tianhao/debug/dsv4/aiconfigurator/collector/deep_collector

env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NVSHMEM_IBGDA_NIC_HANDLER=cpu \
  NVSHMEM_IB_GID_INDEX=3 \
  NVSHMEM_IB_ROCE_VERSION_NUM=2 \
  NVSHMEM_IB_ADDR_FAMILY=AF_INET \
  NVSHMEM_IB_ADDR_RANGE=172.18.0.0/19 \
  NVSHMEM_ENABLE_NIC_PE_MAPPING=1 \
  NVSHMEM_HCA_LIST=mlx5_0:1 \
  NCCL_IB_GID_INDEX=3 \
  python test_low_latency.py \
    --num-processes 8 \
    --num-tokens 4 \
    --hidden 7168 \
    --num-topk 6 \
    --num-experts 384
```

Communication results:

| source | shape | dispatch | combine | total | note |
|---|---|---:|---:|---:|---|
| AIC H200 DB | t4 h7168 topk8 e256 | 15.750 us | 14.054 us | 29.804 us | nearest stored single-node row |
| local AIC collector | t4 h7168 topk8 e256 | 12.650 us | 12.990 us | 25.640 us | same reference shape, H20 local |
| local AIC collector | t4 h7168 topk6 e384 | 13.970 us | 13.040 us | 27.010 us | exact DSV4 local dispatch shape |
| local AIC collector + MNNVL | t4 h7168 topk6 e384 | 14.200 us | 12.930 us | 27.130 us | no material change |
| EP8 nsys trace rank6 | t4 h7168 topk6 e384 | 297.343 us | 387.295 us | 684.638 us | long module-attributed kernel residency |

Interpretation:

- For AIC-style manual communication, use about `27 us` for this DSV4 decode
  shape on the local H20 environment.
- `mock full-layer compute + AIC-style comm = 1425.277 + 27.010 = 1452.287 us`.
  This lines up with the EP8 compute-kernel rank-max (`1452.285 us`), but not
  with EP8 all-kernel rank-max when long communication residency is counted.
- The EP8 nsys `dispatch/combine` rows include long kernel residency or waits
  inside the SGLang `mlp.experts` range. They should not be mixed with the AIC
  DeepEP collector's active dispatch/combine latency unless the goal is to model
  wall-critical wait behavior rather than AIC op semantics.

Files:

- `/tianhao/debug/dsv4/experiments_20260426/exp9_deepep_comm_aic_collector/README.md`
- `/tianhao/debug/dsv4/experiments_20260426/exp9_deepep_comm_aic_collector/summary.csv`

## AIC DeepGEMM Baseline

The comparison copy is under:

```text
../comparison/collect_moe_deepgemm.py
```

It is the AIC op-level DeepGEMM baseline kept in this repo for comparison. Its
default rank selection is aligned with AIC `collect_moe.py` rank0-local
semantics. It constructs the post-DeepEP-dispatch masked DeepGEMM runner input
and measures only routed expert compute. It does not call
`deep_ep.Buffer.low_latency_dispatch` or `low_latency_combine`, so it is not a
communication collector. When run from this repo, provide the AIC collector path
on `PYTHONPATH` so it can import `common_test_cases` and `helper`:

```bash
PYTHONPATH=/tianhao/debug/dsv4/aiconfigurator/collector:/workspace/sglang/python \
COLLECTOR_DEEPGEMM_MOE_NUM_TOKENS=8 \
COLLECTOR_DEEPGEMM_MOE_TP=1 \
COLLECTOR_DEEPGEMM_MOE_EP=8 \
COLLECTOR_DEEPGEMM_MOE_DISTRIBUTION=power_law \
COLLECTOR_DEEPGEMM_MOE_MAX_CASES=1 \
COLLECTOR_DEEPGEMM_MOE_RANK=0 \
COLLECTOR_DEEPGEMM_MOE_NUM_RUNS=20 \
python ../comparison/collect_moe_deepgemm.py
```

By default it writes `moe_perf.txt` in the current working directory.

## Interpreting Results

For DeepGEMM routed expert compute, compare these quantities:

```text
AIC baseline:  kernel_time_us_per_iter or CSV latency from ../comparison
full-layer:    step/module/kernel breakdown from single-GPU mock and EP8 traces
```

For full-layer MLP, compare:

```text
gate + topk + shared_experts + experts
```

For DeepEP communication, compare:

```text
AIC/DeepEP active comm: test_low_latency.py dispatch/combine latency
EP8 nsys comm rows:     module-attributed dispatch/combine kernel residency
```

Do not mix op-level and full-layer boundaries in one table. The single-GPU
standard path's `mlp.experts` includes local reorder/preprocess/post-reorder.
The DeepEP low-latency path's `mlp.experts` is much closer to DeepGEMM core
compute, because dispatch already produced the masked DeepGEMM input tensors.
Similarly, do not use `collect_moe_deepgemm.py` as a communication baseline; use
it only for routed expert compute.

## Cleanup Notes

Generated files usually live under `/tmp`:

```bash
rm -f /tmp/dsv4_*_nsys.nsys-rep /tmp/dsv4_*_nsys.sqlite
```

Do not delete traces before extracting step/module breakdowns if the exact
comparison needs to be reproduced.
