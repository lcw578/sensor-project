# -*- coding: utf-8 -*-
"""决定性分析：测试集中"存在近重复训练样本"的部分 vs "干净"部分，准确率差多少。"""
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(_ROOT, "outputs", "cache")
Xg = np.load(f"{CACHE}/X_glove_raw.npy").reshape(2880, -1).astype(np.float64)
y = np.load(f"{CACHE}/y.npy")
meta = pd.read_csv(f"{CACHE}/meta.csv")
split = meta["split"].values
tr, te = split == "train", split == "test"
te_idx = np.where(te)[0]                 # 测试集下标（升序，与 test_pred.csv 顺序一致）


def l2norm(X):
    X = X - X.mean(1, keepdims=True)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


Z = l2norm(Xg)
sim = Z[te] @ Z[tr].T                    # (288, 2304) 测试 x 训练
nn = sim.argmax(axis=1)                  # 最近邻在训练集中的位置
mx = sim.max(axis=1)
tr_idx = np.where(tr)[0]
nn_label = y[tr_idx[nn]]
same_cls = nn_label == y[te]

print("=" * 74)
print("[A] 近重复检测：每个测试样本在训练集中的最近邻")
print("=" * 74)
print(f"  最近邻相似度: 均值 {mx.mean():.4f}  中位 {np.median(mx):.4f}  最大 {mx.max():.4f}")
print(f"  最近邻与测试样本同类别的比例 = {same_cls.mean()*100:.1f}%  "
      f"(若 <100% 说明存在跨类近邻 → 标签可疑)")
for thr in [0.999, 0.995, 0.99, 0.95]:
    m = mx >= thr
    print(f"  相似度 >= {thr}: {m.sum():3d}/288 ({m.mean()*100:5.1f}%)  "
          f"其中同类 {int((m & same_cls).sum())} / 异类 {int((m & ~same_cls).sum())}")

print("\n" + "=" * 74)
print("[B] 决定性检验：按'是否有近重复'分层看测试准确率（基线 bilstm_both）")
print("=" * 74)
pred = pd.read_csv(f"{CACHE}/../results/bilstm_both_test_pred.csv")
assert len(pred) == 288, len(pred)
ok = (pred["y_true"].values == pred["y_pred"].values)
# 校验顺序一致性
assert (pred["y_true"].values == y[te_idx]).all(), "test_pred 顺序与 split 不一致"

for thr in [0.999, 0.99, 0.95]:
    dup = mx >= thr
    print(f"\n  阈值 {thr}:")
    if dup.sum() > 0:
        print(f"    有近重复的测试样本 n={dup.sum():3d}  准确率 = {ok[dup].mean()*100:6.2f}%")
    if (~dup).sum() > 0:
        print(f"    无近重复的测试样本 n={(~dup).sum():3d}  准确率 = {ok[~dup].mean()*100:6.2f}%")

print("\n" + "=" * 74)
print("[C] 若按相似度分档，各档准确率")
print("=" * 74)
bins = [(0.999, 1.01, ">=0.999 (近乎重复)"), (0.99, 0.999, "0.99~0.999"),
        (0.95, 0.99, "0.95~0.99"), (0.0, 0.95, "<0.95 (最难)")]
for lo, hi, nm in bins:
    m = (mx >= lo) & (mx < hi)
    if m.sum():
        print(f"  {nm:22s} n={m.sum():3d}  准确率 = {ok[m].mean()*100:6.2f}%")

print("\n" + "=" * 74)
print("[D] 对比：sEMG 模态下的同一检验（用 modality_semg 的预测）")
print("=" * 74)
Xs = np.load(f"{CACHE}/X_semg_raw.npy").reshape(2880, -1).astype(np.float64)
Zs = l2norm(Xs)
sims = Zs[te] @ Zs[tr].T
mxs = sims.max(axis=1)
preds = pd.read_csv(f"{CACHE}/../results/modality_semg_test_pred.csv")
oks = (preds["y_true"].values == preds["y_pred"].values)
for thr in [0.999, 0.99]:
    dup = mxs >= thr
    if dup.sum():
        print(f"  相似度>={thr}: 有近重复 n={dup.sum():3d} acc={oks[dup].mean()*100:6.2f}%  |  "
              f"无近重复 n={(~dup).sum():3d} acc={oks[~dup].mean()*100:6.2f}%")
