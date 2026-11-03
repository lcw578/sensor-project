# -*- coding: utf-8 -*-
"""全量扫描 24 手语数据集：结构一致性 / 零通道 / 异常值 / 每类统计。
用法: python scan_dataset.py <data_root>
"""
import os
import sys
import pickle
import numpy as np
from collections import defaultdict

DATA_ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "24手语数据", "data")
KEYS = ["res1", "res2", "muscle1", "muscle2"]
CLASSES = sorted(os.listdir(DATA_ROOT))

shape_counter = defaultdict(int)
zero_chan = {k: None for k in KEYS}      # 每个 key：逐通道“是否在所有样本中恒为 0”
nan_ok = 0
nan_bad = []
global_min = {k: np.inf for k in KEYS}
global_max = {k: -np.inf for k in KEYS}
per_class_n = {}
per_class_chan_std = {k: [] for k in KEYS}

total = 0
for ci, cls in enumerate(CLASSES):
    cdir = os.path.join(DATA_ROOT, cls)
    if not os.path.isdir(cdir):
        continue
    files = [f for f in os.listdir(cdir) if f.endswith(".pkl")]
    files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    per_class_n[cls] = len(files)
    for fn in files:
        total += 1
        try:
            d = pickle.load(open(os.path.join(cdir, fn), "rb"))
        except Exception as e:
            nan_bad.append((cls, fn, "LOAD_FAIL:" + repr(e)))
            continue
        for k in KEYS:
            a = np.asarray(d[k])
            shape_counter[(k, a.shape)] += 1
            if np.isnan(a).any() or np.isinf(a).any():
                nan_bad.append((cls, fn, k))
            global_min[k] = min(global_min[k], float(a.min()))
            global_max[k] = max(global_max[k], float(a.max()))
            ch_zero = np.all(a == 0, axis=1)     # 每个通道是否整段为 0
            if zero_chan[k] is None:
                zero_chan[k] = ch_zero.copy()
            else:
                zero_chan[k] &= ch_zero
            # 累积通道级 std 用于观察活跃度
            per_class_chan_std[k].append(a.std(axis=1))

print("=" * 70)
print(f"数据根目录: {DATA_ROOT}")
print(f"类别数: {len(per_class_n)}  总样本数: {total}")
print("=" * 70)
print("\n[每类样本数]")
for c in CLASSES:
    print(f"  {c:14s}: {per_class_n.get(c, 0)}")

print("\n[各 key 出现过的 shape]")
for (k, sh), n in sorted(shape_counter.items()):
    print(f"  {k:8s} {str(sh):14s} x{n}")

print("\n[恒定全零通道（在所有样本中恒为 0）] —— 索引 0-based")
for k in KEYS:
    if zero_chan[k] is not None:
        idx = [i for i, z in enumerate(zero_chan[k]) if z]
        print(f"  {k:8s} 恒定零通道 {len(idx)}/{len(zero_chan[k])} : {idx}")

print("\n[全局数值范围]")
for k in KEYS:
    print(f"  {k:8s} min={global_min[k]:.4f}  max={global_max[k]:.4f}")

print("\n[NaN/Inf 或读取失败样本] 数量:", len(nan_bad))
for item in nan_bad[:20]:
    print("  ", item)

print("\n[每类通道平均 std (0-based 通道均值)]")
for k in KEYS:
    arr = np.array(per_class_chan_std[k])   # (样本数, 通道数)
    print(f"  {k:8s} 全体通道 std 均值: {arr.mean(axis=0).round(4)}")
