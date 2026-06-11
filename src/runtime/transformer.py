"""Compatibility alias for the DeepSeek-V4 model implementation.

The implementation lives in :mod:`src.models.deepseek_v4.transformer`.
This alias keeps existing ``src.runtime.transformer`` imports bound to the
same module object so mutable module globals (rank/world_size/dtype state)
remain shared.
"""

from __future__ import annotations

import sys as _sys

from src.models.deepseek_v4 import transformer as _impl

_sys.modules[__name__] = _impl
