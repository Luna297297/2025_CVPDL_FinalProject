"""YOLO12 with ShiftWise Convolution - Independent Project

This package extends Ultralytics YOLO with ShiftWise convolution modules
without modifying the original ultralytics source code.

Usage:
    from yolo12_shiftwise import apply_shiftwise_patch
    apply_shiftwise_patch()  # Inject ShiftWise modules into ultralytics
    
    from ultralytics import YOLO
    model = YOLO("yolo12s_shiftwise.yaml")
"""

from .patches import apply_shiftwise_patch

__version__ = "0.1.0"
__all__ = ["apply_shiftwise_patch"]

