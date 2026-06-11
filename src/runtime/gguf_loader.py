"""Compatibility alias for the DeepSeek-V4 GGUF runtime loader."""

from __future__ import annotations

import sys as _sys

from src.models.deepseek_v4 import gguf_loader as _impl

_sys.modules[__name__] = _impl
