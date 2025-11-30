"""ShiftWise block modules for YOLO."""

from __future__ import annotations

import torch
import torch.nn as nn


class BottleneckSW(nn.Module):
    """Bottleneck variant that uses ShiftWiseConv for large receptive field.
    
    This replaces BOTH conv layers with ShiftWiseConv to achieve large kernel
    effects as per the ShiftWise paper. The big_k parameter controls the
    equivalent large kernel size (M in paper, typically 13-51).
    
    Args:
        c1: Input channels
        c2: Output channels
        shortcut: Whether to use shortcut connection
        e: Expansion ratio
        big_k: Equivalent large kernel size for ShiftWise (must be >> 3)
        replace_both: If True, replace both cv1 and cv2 with ShiftWiseConv.
                     If False, only replace cv2 (backward compatibility).
    """

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, big_k: int = 13, replace_both: bool = True
    ):
        """Initialize a ShiftWise bottleneck with configurable big_k."""
        super().__init__()
        
        # 動態導入（避免循環依賴）
        from ultralytics.nn.modules.conv import Conv
        from .shiftwise import ShiftWiseConv
        
        c_ = int(c2 * e)
        
        if replace_both:
            # Replace both layers with ShiftWiseConv (target architecture)
            self.cv1 = ShiftWiseConv(c1, c_, big_k=big_k, small_k=3, s=1)
            self.cv2 = ShiftWiseConv(c_, c2, big_k=big_k, small_k=3, s=1)
        else:
            # Only replace cv2 (backward compatibility)
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = ShiftWiseConv(c_, c2, big_k=big_k, small_k=3, s=1)
        
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ShiftWise bottleneck."""
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


def _create_c3k2_sw_class():
    """動態創建 C3k2_SW 類，繼承自 C2f（避免循環依賴）"""
    from ultralytics.nn.modules.block import C2f, C3k
    
    class C3k2_SW(C2f):
        """C3k2 variant backed by ShiftWise bottlenecks with configurable big_k.
        
        This module allows per-stage configuration of big_k (equivalent large kernel size)
        as per the ShiftWise paper's multi-stage design. Different stages should use
        different big_k values (e.g., 13, 17, 21, 25) to achieve hierarchical receptive fields.
        
        Args:
            c1: Input channels
            c2: Output channels
            n: Number of bottleneck blocks
            c3k: Whether to use C3k blocks (if True, falls back to standard C3k)
            e: Expansion ratio
            g: Groups for convolutions
            shortcut: Whether to use shortcut connections
            big_k: Equivalent large kernel size for ShiftWise (must be >> 3, paper uses 13-51)
            replace_both: If True, replace both conv layers in each bottleneck with ShiftWiseConv
        """

        def __init__(
            self,
            c1: int,
            c2: int,
            n: int = 1,
            c3k: bool = False,
            e: float = 0.5,
            g: int = 1,
            shortcut: bool = True,
            big_k: int = 13,
            replace_both: bool = True,
        ):
            """Initialize ShiftWise-enabled C3k2 module with configurable big_k.
            
            Args:
                c1: Input channels
                c2: Output channels
                n: Number of bottleneck blocks
                c3k: Whether to use C3k blocks (if True, falls back to standard C3k)
                e: Expansion ratio
                g: Groups for convolutions
                shortcut: Whether to use shortcut connections
                big_k: Equivalent large kernel size for ShiftWise (must be >> 3, paper uses 13-51)
                replace_both: If True, replace both conv layers in each bottleneck with ShiftWiseConv
            """
            # 調用父類 C2f 的 __init__
            # C2f 的參數順序是: (c1, c2, n, shortcut, g, e)
            # parse_model 傳入的參數順序是: (c1, c2, n, c3k, e, g, shortcut, big_k, replace_both)
            # 所以我們需要重新排列參數並確保類型正確
            import inspect
            frame = inspect.currentframe()
            args_values = inspect.getargvalues(frame)
            
            # 確保所有參數都是正確的類型（避免 tuple 或其他類型錯誤）
            c1 = int(c1) if not isinstance(c1, (tuple, list)) else int(c1[0] if isinstance(c1, (tuple, list)) else c1)
            c2 = int(c2) if not isinstance(c2, (tuple, list)) else int(c2[0] if isinstance(c2, (tuple, list)) else c2)
            n = int(n) if not isinstance(n, (tuple, list)) else int(n[0] if isinstance(n, (tuple, list)) else n)
            
            # 確保 e, g, shortcut 是正確類型
            if isinstance(e, (tuple, list)):
                e = float(e[0]) if len(e) > 0 else 0.5
            else:
                e = float(e) if e is not None else 0.5
                
            if isinstance(g, (tuple, list)):
                g = int(g[0]) if len(g) > 0 else 1
            else:
                g = int(g) if g is not None else 1
                
            if isinstance(shortcut, (tuple, list)):
                shortcut = bool(shortcut[0]) if len(shortcut) > 0 else True
            else:
                shortcut = bool(shortcut) if shortcut is not None else True
            
            super().__init__(c1, c2, n, shortcut, g, e)
            
            # 替換 m 為 ShiftWise 版本
            if c3k:
                block = C3k
                self.m = nn.ModuleList(block(self.c, self.c, 2, shortcut, g) for _ in range(n))
            else:
                # Use BottleneckSW with configurable big_k
                # 注意：這裡需要從當前模組導入，避免循環依賴
                import sys
                current_module = sys.modules[__name__]
                BottleneckSW = getattr(current_module, 'BottleneckSW', None)
                if BottleneckSW is None:
                    # 如果還沒定義，從外部導入
                    from .block import BottleneckSW
                self.m = nn.ModuleList(
                    BottleneckSW(self.c, self.c, shortcut, e=1.0, big_k=big_k, replace_both=replace_both)
                    for _ in range(n)
                )
    
    return C3k2_SW


# 延遲創建類（在第一次使用時）
_C3k2_SW_class = None


def get_c3k2_sw_class():
    """獲取 C3k2_SW 類（延遲加載）"""
    global _C3k2_SW_class
    if _C3k2_SW_class is None:
        _C3k2_SW_class = _create_c3k2_sw_class()
    return _C3k2_SW_class


# 為了向後相容，創建一個代理類
class C3k2_SW:
    """代理類，延遲加載實際的 C3k2_SW 實現"""
    
    def __new__(cls, *args, **kwargs):
        actual_class = get_c3k2_sw_class()
        return actual_class(*args, **kwargs)

