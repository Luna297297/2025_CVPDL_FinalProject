"""Patch system to inject ShiftWise modules into ultralytics without modifying source code."""

from .tasks import apply_shiftwise_patch

__all__ = ["apply_shiftwise_patch"]

