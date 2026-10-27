import atexit
import cProfile
import os
import pstats

from vllm_fl.worker.model_runner import ModelRunnerFL as GPUModelRunner


_OUTPUT = os.environ["HOST_PROFILE"]
_INTERVAL = int(os.environ.get("HOST_PROFILE_INTERVAL", "100"))
_profiler = cProfile.Profile()
_calls = 0


def _dump():
    if _calls == 0:
        return
    _profiler.dump_stats(_OUTPUT + ".prof")
    with open(_OUTPUT + ".txt", "w", encoding="utf-8") as f:
        stats = pstats.Stats(_profiler, stream=f)
        stats.sort_stats("cumulative").print_stats(120)


def _execute_model(self, *args, **kwargs):
    global _calls
    _profiler.enable()
    try:
        return _original(self, *args, **kwargs)
    finally:
        _profiler.disable()
        _calls += 1
        if _calls % _INTERVAL == 0:
            _dump()


_original = GPUModelRunner.execute_model
GPUModelRunner.execute_model = _execute_model
atexit.register(_dump)

import runpy

if __name__ == "__main__":
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
