# -*- coding: utf-8 -*-
"""泄漏审计 + 与论文差异的归因分析（只读）。"""
import os
import json
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(_ROOT, "outputs", "cache")
Xg = np.load(f"{CACHE}/X_glove_raw.npy").reshape(2880, -1).astype(np.float64)
Xs = np.load(f"{CACHE}/X_semg_raw.npy").reshape(2880, -1).astype(np.float64)
y = np.load(f"{CACHE}/y.npy")
meta = pd.read_csv(f"{CACHE}/meta.csv")
split = meta["split"].values
tr, va, te = split == "train", split == "val", split == "test"

print("=" * 74)
print("[1] 划分完整性")
print("=" * 74)
print(f"  样本数: train={tr.sum()} val={va.sum()} test={te.sum()} 合计={tr.sum()+va.sum()+te.sum()}")
S = [set(np.where(tr)[0]), set(np.where(va)[0]), set(np.where(te)[0])]
print(f"  train∩val={len(S[0]&S[1])}  train∩test={len(S[0]&S[2])}  val∩test={len(S[1]&S[2])}  "
      f"(全为 0 才算无重叠)")
print(f"  下标并集大小 = {len(S[0]|S[1]|S[2])} (应 = 2880)")
print(f"  meta 的 label 与 y.npy 是否一致: {bool((meta['label'].values == y).all())}")

print("\n  每类在三集合中的数量（应各为 96/12/12）:")
cnt = pd.crosstab(meta["class"], meta["split"])
bad = cnt[(cnt.get("train", 0) != 96) | (cnt.get("val", 0) != 12) | (cnt.get("test", 0) != 12)]
print(f"    不符合的类别数 = {len(bad)} (应为 0)")


def l2norm(X):
    X = X - X.mean(1, keepdims=True)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


print("\n" + "=" * 74)
print("[2] 重复样本泄漏检测：每个测试样本在训练集中的最大余弦相似度")
print("=" * 74)
for nm, X in [("glove", Xg), ("sEMG", Xs)]:
    Z = l2norm(X)
    Ztr, Zte = Z[tr], Z[te]
    sim = Zte @ Ztr.T                      # (288, 2304)
    mx = sim.max(axis=1)
    print(f"  {nm:6s} 测试样本对训练集的最大相似度: 均值 {mx.mean():.3f}  中位 {np.median(mx):.3f}  "
          f"最大 {mx.max():.3f}")
    print(f"         >0.99 的测试样本数 = {(mx > 0.99).sum()}  >0.95 = {(mx > 0.95).sum()}  "
          f"（若存在≈1.0 说明有重复样本泄漏）")

print("\n" + "=" * 74)
print("[3] 同一样本被'复制'到不同 split 的情况（同文件内容哈希）")
print("=" * 74)
import pickle, hashlib
root = os.path.join(_ROOT, "24手语数据", "data")
classes = sorted(os.listdir(root))
h = {}
dup = 0
for ci, cls in enumerate(classes):
    for fn in sorted(os.listdir(os.path.join(root, cls)), key=lambda x: int(os.path.splitext(x)[0])):
        with open(os.path.join(root, cls, fn), "rb") as f:
            key = hashlib.md5(f.read()).hexdigest()
        h.setdefault(key, []).append((ci, fn))
for k, v in h.items():
    if len(v) > 1:
        dup += 1
        if dup <= 5:
            print("  重复内容:", v)
print(f"  内容完全相同的 .pkl 组数 = {dup}")

print("\n" + "=" * 74)
print("[4] 会话漂移检测：类均值随样本序号的变化（检验分块划分是否会更难）")
print("=" * 74)
A = l2norm(Xg)
for c in [0, 2, 19, 23]:
    idx = np.where(y == c)[0]
    q = np.array_split(idx, 4)
    means = [l2norm(Xg[idx_]).mean(0) for idx_ in q]
    cent = l2norm(Xg[idx]).mean(0)
    # 各四分位块与类中心的相似度
    sims = [float(m @ (cent / (np.linalg.norm(cent) + 1e-9))) for m in means]
    print(f"  {classes[c]:12s} 四个四分位块与类中心相似度: " +
          " ".join(f"{s:.3f}" for s in sims))

print("\n" + "=" * 74)
print("[5] 统计显著性：288 个测试样本下，模型间差异是否显著")
print("=" * 74)


def ci95(p, n=288):
    se = np.sqrt(p * (1 - p) / n)
    return 1.96 * se * 100


for nm, acc in [("Bi-LSTM", 0.9896), ("1D-CNN", 0.9931), ("TCN", 0.9965), ("Bi-GRU", 0.9861)]:
    k = round(acc * 288)
    print(f"  {nm:9s} acc={acc:.4f} ({k}/288)  95%CI = ±{ci95(acc):.2f} pp")
print("\n  结论：上述模型的置信区间高度重叠 → 单次 288 样本的差异多在噪声范围内。")
