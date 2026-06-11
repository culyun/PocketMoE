"""Compatibility alias for the DeepSeek-V4 partition policy implementation."""

from __future__ import annotations

import sys as _sys

from src.models.deepseek_v4 import partition_policy as _impl

_sys.modules[__name__] = _impl
