"""vLLM general_plugins entry for layerwise benchmarking.

Registered via pyproject.toml as `vllm.general_plugins: layerwise → install`.
vLLM calls `install()` exactly once in every process that loads the plugin
list — engine process AND every TP worker (`WorkerWrapperBase.init_worker`
in `v1/worker/worker_base.py`). This fires under both fork and spawn
multiprocessing, which is the reason this package exists.

`install()` just side-effect-imports the two patch modules living at
`../../vllm/`; their own top-level `_install()` does the actual monkey-patch
on `GPUModelRunner.load_model` and `GPUModelRunner.execute_model`.
"""
import logging
import os
import sys

logger = logging.getLogger(__name__)


def install():
    here = os.path.dirname(os.path.abspath(__file__))
    patches_dir = os.path.normpath(os.path.join(here, "..", "..", "vllm"))
    if patches_dir not in sys.path:
        sys.path.insert(0, patches_dir)
    import vllm_layer_skip_patch  # noqa: F401
    import vllm_step_marker  # noqa: F401
    logger.warning(
        f"[layerwise-plugin] installed in pid={os.getpid()} "
        f"(patches_dir={patches_dir})"
    )
