"""Wrapper: import patches, then run sglang.bench_one_batch.main().

Extras:
  - `--config-overrides '{"text_config.num_hidden_layers": 4}'` to patch
    HF config.json on disk before load (avoids `config.update()` shallow
    overwrite of nested dicts; supports dotted keys).

Steps:
  1. Import `sglang_model_patches` — installs model-behavior monkey-patches.
  2. Pre-parse `--config-overrides`; if present, download + patch model config,
     rewrite `--model-path` in argv to the local tmp dir.
  3. Import sglang.bench_one_batch, force mp fork, call main().

Usage:
    LAYERWISE_TARGET_LAYERS=3 python run_bench_skip.py \\
        --model-path Qwen/Qwen3.5-122B-A10B-FP8 --trust-remote-code --load-format dummy \\
        --config-overrides '{"text_config.num_hidden_layers": 4}' \\
        --tp-size 1 --enable-layerwise-nvtx-marker \\
        --batch-size 1 --input-len 512 --output-len 3
"""
import argparse
import json
import logging
import multiprocessing as mp
import os
import sys

# Make sibling dirs importable.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)  # local SGLang profiling patches
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "common"))  # config_patch

import sglang_model_patches  # noqa: F401  (model behavior patches)
import moe_topk_hack_patch  # noqa: F401  (optional MoE power-law topk ids)
import moe_ep_local_experts_patch  # noqa: F401  (optional full-layer EP-local experts timing)
import sglang_step_marker  # noqa: F401  (optional bench_step::* NVTX ranges)
from config_patch import patch_model_path

from sglang.bench_one_batch import BenchArgs, main
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree

mp.set_start_method("fork", force=True)


def _maybe_patch_model_path(argv):
    """Pre-parse --config-overrides / --model-path; rewrite model-path to a
    local tmp dir with patched config.json if overrides provided. Returns
    possibly-modified argv list.
    """
    i = 0
    overrides_json = None
    model_path = None
    while i < len(argv):
        if argv[i] == "--config-overrides" and i + 1 < len(argv):
            overrides_json = argv[i + 1]
            # Remove this flag; not passed to sglang
            del argv[i : i + 2]
            continue
        if argv[i] == "--model-path" and i + 1 < len(argv):
            model_path = argv[i + 1]
        i += 1

    if overrides_json and model_path:
        overrides = json.loads(overrides_json)
        # Auto-select model_type rewrites + auto_map strip for known quirks.
        rewrites = {
            "glm_moe_dsa": "deepseek_v3",
            "deepseek_v32": "deepseek_v3",
            # Transformers does not know DeepSeek-V4 yet. Rewriting to
            # deepseek_ref lets sglang's DeepSeek-V4 config loader take over
            # while keeping architectures=["DeepseekV4ForCausalLM"] intact.
            "deepseek_v4": "deepseek_ref",
        }
        local = patch_model_path(model_path, overrides,
                                 strip_auto_map=True,
                                 model_type_rewrites=rewrites)
        print(f"[config-patch] model_path {model_path} -> {local}")
        # Rewrite --model-path in argv
        for j in range(len(argv)):
            if argv[j] == "--model-path":
                argv[j + 1] = local
                break
    return argv


if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + _maybe_patch_model_path(sys.argv[1:])

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        main(server_args, bench_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)
