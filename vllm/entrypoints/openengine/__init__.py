# SPDX-License-Identifier: Apache-2.0
"""OpenEngine v1 gRPC server for vLLM.

Wraps a vLLM ``AsyncLLM`` (via the ``EngineClient`` ABC) behind the
vendor-neutral OpenEngine v1 contract so a Dynamo sidecar worker can drive the
engine across a process boundary. See CLAUDE.md in this directory for design.
"""

from .config import EngineServeConfig
from .server import OpenEngineServer, build_serve_config
from .servicer import OpenEngineServicer

__all__ = [
    "OpenEngineServer",
    "OpenEngineServicer",
    "EngineServeConfig",
    "build_serve_config",
]
