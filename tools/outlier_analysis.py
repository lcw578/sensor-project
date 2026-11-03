# -*- coding: utf-8 -*-
"""定位数值离群样本 + 类间相似度（Pearson）分析。
用法: python outlier_analysis.py <data_root>
"""
import os
import sys
import pickle
import numpy as np

DATA_ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "24手语数据", "data")
CLASSES = sorted(os.listdir(DATA_ROOT))
KEYS = ["res1", "res2", "muscle1", "muscle2"]

# ---- 1. 定位极端幅值样本 ----
records = []
means = {k: [] for k in KEYS}
labels = []
for cls in CLASSES:
    cdir = os.path.join(DATA_ROOT, cls)
    files = sorted([f for f in os.listdir(cdir) if f.endswith(".pkl")],
                   key=lambda x: int(os.path.splitext(x)[0]))
    for fn in files:
        d = pickle.load(open(os.path.join(cdir, fn), "rb"))
        row = {"cls": cls, "file": fn}
        for k in KEYS:
            a = np.asarray(d[k], dtype=np.float64)
            row[k + "_absmax"] = float(np.abs(a).max())
            row[k + "_std"] = float(a.std())
            means[k].append(a.mean(axis=1))
        records.append(row)
        labels.append(cls)

print("=" * 78)
print("[各 key 绝对幅值 Top-8 样本]")
for k in KEYS:
    top = sorted(records, key=lambda r: r[k + "_absmax"], reverse=True)[:8]
    print(f"\n  --- {k} ---")
    for r in top:
        print(f"    {r['cls']:14s} {r['file']:>8s}  absmax={r[k+'_absmax']:12.2f}  std={r[k+'_std']:9.3f}")

# ---- 2. 类中心向量 + 类间 Pearson 相似度 ----
print("\n" + "=" * 78)
print("[类间相似度] 用每类 120 样本的通道均值向量(96 维: 20+20+10+10)做类中心，计算两两 Pearson r")
feat = np.array([np.concatenate([means[k][i] for k in KEYS]) for i in range(len(labels))])
cls_idx = {c: np.where(np.array(labels) == c)[0] for c in CLASSES}
centers = np.array([feat[idx].mean(axis=0) for c in CLASSES for idx in [cls_idx[c]]])
C = (centers - centers.mean(axis=1, keepdims=True))
C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
corr = C @ C.T
np.fill_diagonal(corr, -1)
print(f"  总体平均类间 r = {corr[corr>-1].mean():.4f}")
flat = [(corr[i, j], CLASSES[i], CLASSES[j]) for i in range(len(CLASSES)) for j in range(i + 1, len(CLASSES))]
flat.sort(reverse=True)
print("  最相似的 10 对类别:")
for r, a, b in flat[:10]:
    print(f"    {a:14s} <-> {b:14s}  r = {r:.4f}")
print("  最不相似的 5 对类别:")
for r, a, b in flat[-5:]:
    print(f"    {a:14s} <-> {b:14s}  r = {r:.4f}")
