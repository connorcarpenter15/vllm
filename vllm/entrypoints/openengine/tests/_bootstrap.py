# SPDX-License-Identifier: Apache-2.0
"""Load the OpenEngine servicer + config + stubs as a standalone package.

The servicer logic is vLLM-free, but it lives under the ``vllm.`` namespace, so
a normal ``import vllm.entrypoints.openengine.servicer`` would execute
``vllm/__init__.py`` (torch, CUDA, ...). To keep the servicer/wire tests
runnable on a CPU box without vLLM installed, we register a synthetic package
whose ``__path__`` points at the openengine dir and import the pure submodules
under it -- without ever importing the real ``vllm`` package.

This bootstrap is a no-op-equivalent when vLLM *is* installed (cluster runs):
it still loads the same source files, just under an alias package name.
"""

from __future__ import annotations

import importlib
import os
import pathlib
import sys
import types

_ALIAS = "_oe_standalone"
# Default to the openengine dir (parent of tests/). A local CPU run can copy the
# test files elsewhere and point here via OE_DIR_OVERRIDE.
_OE_DIR = pathlib.Path(
    os.environ.get(
        "OE_DIR_OVERRIDE", str(pathlib.Path(__file__).resolve().parent.parent)
    )
)


def load():
    """Return ``(servicer_module, pb_module)`` loaded standalone."""
    if _ALIAS not in sys.modules:
        pkg = types.ModuleType(_ALIAS)
        pkg.__path__ = [str(_OE_DIR)]
        pkg.__package__ = _ALIAS
        sys.modules[_ALIAS] = pkg

    servicer = importlib.import_module(f"{_ALIAS}.servicer")
    pb = importlib.import_module(f"{_ALIAS}._openengine.openengine_pb2")
    return servicer, pb
