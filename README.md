## 專案結構

```
yolo12-shiftwise/
├── yolo12_shiftwise/
│   ├── __init__.py
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── shiftwise.py      # ShiftWiseConv 模組
│   │   └── block.py           # BottleneckSW, C3k2_SW
│   └── patches/
│       ├── __init__.py
│       └── tasks.py            # 動態注入到 ultralytics
├── configs/                    # YAML 配置文件
├── scripts/                    # 測試腳本
├── setup.py
├── requirements.txt
├── README.md
└── install.sh
```

## 安裝

### 1. 安裝依賴

```bash
pip install ultralytics
```

### 2. 安裝 shift-wiseConv（CUDA 擴展）

```bash
git clone https://github.com/lidc54/shift-wiseConv.git
cd shift-wiseConv/shiftadd
python setup.py install
```

### 3. 安裝 yolo12-shiftwise

```bash
cd yolo12-shiftwise
pip install -e .
```

## 使用方法

### 基本使用

```python
# 1. 應用 ShiftWise patch（注入模組到 ultralytics）
from yolo12_shiftwise import apply_shiftwise_patch
apply_shiftwise_patch()

# 2. 正常使用 ultralytics
from ultralytics import YOLO

# 3. 載入包含 C3k2_SW 的 YAML 配置
model = YOLO("configs/yolo12s_shiftwise.yaml")

# 4. 訓練
```

### YAML 配置範例

```yaml
# configs/yolo12s_shiftwise.yaml
backbone:
  - [-1, 2, C3k2_SW, [256, False, 0.25]]  # 使用 ShiftWise
```

## 工作原理

1. **Monkey Patching**：`apply_shiftwise_patch()` 函數會：
   - 將 `ShiftWiseConv`, `BottleneckSW`, `C3k2_SW` 注入到 `ultralytics.nn.modules` 命名空間
   - 修改 `parse_model` 函數以識別 `C3k2_SW`
   - 將 `C3k2_SW` 註冊到 `base_modules` 和 `repeat_modules`

2. **動態導入**：ShiftWise 模組會動態導入 ultralytics 的基礎模組（如 `Conv`, `C2f`），避免循環依賴。

3. **無侵入性**：不修改 ultralytics 源碼，可以隨時更新 ultralytics。


## 故障排除

### CUDA 版本不相容

如果遇到 CUDA illegal memory access 錯誤，請：
1. 重新編譯 shift-wiseConv（使用當前 CUDA 版本）
2. 確保記憶體連續性（已內建在代碼中）
3. 如果仍失敗，考慮降級 PyTorch

### 模組未找到

確保已執行 `apply_shiftwise_patch()`：

```python
from yolo12_shiftwise import apply_shiftwise_patch
apply_shiftwise_patch()  # 必須先執行這個
```

## 授權

本專案遵循 AGPL-3.0 授權（與 ultralytics 相同）。

## 參考

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [ShiftWise Paper](https://arxiv.org/abs/2401.12736)
- [ShiftWise Implementation](https://github.com/lidc54/shift-wiseConv)

