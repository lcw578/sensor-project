# -*- coding: utf-8 -*-
"""
阶段一：数据读取与预处理流水线
==============================

输入:  <data_root>/<类别名>/<1..120>.pkl
       每个 pkl = dict{ 'res1'(20,400), 'res2'(20,400), 'muscle1'(10,1950), 'muscle2'(10,1950) }  float32

处理流程（详见 README / 报告）:
  1. 逐文件读取，把同一模态的左右手在【通道维】拼接:
        glove = concat([res1, res2], axis=0).T   -> (400, 40)
        semg  = concat([muscle1, muscle2], axis=0).T -> (1950, 20)
     （对齐源论文的 400x40 / 1950x20，时间主序）
  2. 标签: 24 个类别名按字典序固定映射为 0..23
  3. 划分: 以 .pkl 文件为单位、按类别分层，8:1:1（每类 96/12/12），固定随机种子
  4. 归一化统计量【只用训练集】计算，避免测试集信息泄漏
  5. 逐通道分位截断（默认 1%~99%）抑制离群尖峰
  6. 逐通道标准化; 恒定通道( std<eps )置零，避免除零产生 NaN

输出:  cache_dir/
        X_glove_raw.npy (N,400,40) f32      X_semg_raw.npy  (N,1950,20) f32
        X_glove.npy     (N,400,40) f32      X_semg.npy      (N,1950,20) f32   （已归一化）
        y.npy (N,) int64                    meta.csv (index,class,file,split)
        label_map.json                      norm_stats.npz
        preprocess_report.json
"""
import os
import sys
import json
import time
import pickle
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.filters import apply_filter  # noqa: E402

# ---------------------------------------------------------------- 常量
KEYS = ["res1", "res2", "muscle1", "muscle2"]
# 默认路径基于仓库根目录（本文件在 <root>/src/data/ 下）；数据集需自行放置到 24手语数据/data/
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_ROOT = os.path.join(_ROOT, "24手语数据", "data")
DEFAULT_CACHE_DIR = os.path.join(_ROOT, "outputs", "cache")


# ---------------------------------------------------------------- 工具
def discover_classes(data_root):
    """返回按字典序排序的类别名列表（保证 label 映射可复现）。"""
    classes = sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    )
    return classes


def load_one(path, glove_keys, semg_keys):
    """读一个 pkl，返回 (glove(400,40), semg(1950,20))。"""
    with open(path, "rb") as f:
        d = pickle.load(f)
    glove = np.concatenate([np.asarray(d[k], dtype=np.float32) for k in glove_keys], axis=0).T  # (400,40)
    semg = np.concatenate([np.asarray(d[k], dtype=np.float32) for k in semg_keys], axis=0).T    # (1950,20)
    return np.ascontiguousarray(glove), np.ascontiguousarray(semg)


def load_dataset(data_root, glove_keys, semg_keys, verbose=True):
    """遍历全部类别与文件，返回 (X_glove, X_semg, y, meta)。"""
    classes = discover_classes(data_root)
    X_glove, X_semg, y, meta = [], [], [], []
    t0 = time.time()
    for ci, cls in enumerate(classes):
        cdir = os.path.join(data_root, cls)
        files = [f for f in os.listdir(cdir) if f.endswith(".pkl")]
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        for fn in files:
            g, s = load_one(os.path.join(cdir, fn), glove_keys, semg_keys)
            X_glove.append(g)
            X_semg.append(s)
            y.append(ci)
            meta.append((ci, cls, fn))
        if verbose:
            print(f"  [{ci + 1:2d}/{len(classes)}] {cls:14s} {len(files):3d} 样本")
    X_glove = np.stack(X_glove).astype(np.float32)
    X_semg = np.stack(X_semg).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)
    if verbose:
        print(f"  读取完成: {len(y)} 样本, 用时 {time.time() - t0:.1f}s")
        print(f"  X_glove {X_glove.shape} {X_glove.dtype} | X_semg {X_semg.shape} {X_semg.dtype}")
    return X_glove, X_semg, y, meta, classes


def strat_split(y, ratio, seed):
    """按类别分层划分，返回 split 数组('train'/'val'/'test')。以样本(=pkl)为单位。"""
    rng = np.random.default_rng(seed)
    n = len(y)
    split = np.empty(n, dtype=object)
    assert abs(sum(ratio) - 1.0) < 1e-9, "划分比例之和必须为 1"
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        idx = rng.permutation(idx)          # 固定种子 -> 可复现
        n_train = int(round(len(idx) * ratio[0]))
        n_val = int(round(len(idx) * ratio[1]))
        split[idx[:n_train]] = "train"
        split[idx[n_train:n_train + n_val]] = "val"
        split[idx[n_train + n_val:]] = "test"
    return split


def compute_norm_stats(X_train, clip_q, dead_std_thr=0.05, near_const_thr=0.01):
    """在训练集上按通道统计: 分位截断界 -> (截断后)mean/std -> 死通道掩码。X: (N,T,C)。

    顺序很重要: 先在原始训练集上求分位截断界, 再对训练集截断后求 mean/std,
    这样标准化后的训练集 std 才接近 1, 且统计量只来自训练集(无泄漏)。

    实测: 有效通道全局 std 在 0.3~93, 而噪声底死通道约 0.016, 故 0.05 可干净区分。
    """
    N, T, C = X_train.shape
    flat = X_train.reshape(-1, C)                       # (N*T, C)
    lo = np.quantile(flat, clip_q[0], axis=0)
    hi = np.quantile(flat, clip_q[1], axis=0)
    flat_c = np.clip(flat, lo, hi)                      # 先截断再统计
    mean = flat_c.mean(axis=0)
    std = flat_c.std(axis=0)
    dead = std < dead_std_thr                           # 噪声底通道 -> 置零
    near_const = std < near_const_thr
    return dict(mean=mean, std=std, lo=lo, hi=hi, dead=dead,
                near_const=near_const, C=C)


def apply_norm(X, st, clip=True, mode="global_train"):
    """应用截断+标准化。X:(N,T,C) -> 同形状 f32。"""
    X = X.astype(np.float32, copy=True)
    if mode == "per_sample":
        # 逐样本逐通道 z-score（对传感器漂移鲁棒，但丢失绝对幅值）
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        sd = np.where(sd < 1e-6, 1.0, sd)
        return ((X - mu) / sd).astype(np.float32)
    # ---- 训练集统计的全局/鲁棒标准化 ----
    if clip:
        X = np.clip(X, st["lo"], st["hi"])
    if mode == "robust_train":
        iqr = st["hi"] - st["lo"]
        iqr = np.where(iqr < 1e-6, 1.0, iqr)
        X = (X - st["mean"]) / iqr
    else:  # global_train
        sd = st["std"].copy()
        sd[st["dead"]] = 1.0                            # 死通道先避免除零
        X = (X - st["mean"]) / sd
    # 死通道统一置 0
    if st["dead"].any():
        X[..., st["dead"]] = 0.0
    return X.astype(np.float32)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="24 手语数据集预处理流水线")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratio", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    ap.add_argument("--clip-q", type=float, nargs=2, default=[0.01, 0.99])
    ap.add_argument("--norm-mode", default="global_train",
                    choices=["global_train", "robust_train", "per_sample"])
    ap.add_argument("--std-eps", dest="dead_std_thr", type=float, default=0.05,
                    help="全局 std 低于此值的通道视为噪声底死通道并置零")
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--no-save-raw", action="store_true")
    # ---- 滤波（阶段一可选步骤，用于预处理对比实验 E1–E5）----
    ap.add_argument("--filter", default="none",
                    choices=["none", "bandpass", "notch", "bandpass+notch"],
                    help="sEMG 滤波模式；依据实测主峰在 48.17Hz 而非 50Hz")
    ap.add_argument("--bp", type=float, nargs=2, default=[20.0, 450.0], help="带通截止 lo hi (Hz)")
    ap.add_argument("--bp-order", type=int, default=4)
    ap.add_argument("--notch-freqs", type=float, nargs="+", default=[48.17, 96.34, 144.51])
    ap.add_argument("--notch-q", type=float, default=30.0)
    ap.add_argument("--out-subdir", default="", help="缓存输出子目录（用于区分预处理变体）")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    print("=" * 72)
    print("阶段一：数据读取与预处理")
    print("=" * 72)

    # 1. 读取
    print("\n[1/5] 读取全部 .pkl ...")
    X_glove, X_semg, y, meta, classes = load_dataset(
        args.data_root, ["res1", "res2"], ["muscle1", "muscle2"])

    # 1.5 滤波（可选）
    if args.filter != "none":
        print(f"\n[1.5/5] sEMG 滤波: mode={args.filter} "
              f"(bandpass {args.bp} order={args.bp_order}, notch {args.notch_freqs} q={args.notch_q}) ...")
        before = X_semg.copy()
        X_semg = apply_filter(X_semg, fs=1000.0, mode=args.filter, bp=tuple(args.bp),
                              order=args.bp_order, notch_freqs=tuple(args.notch_freqs),
                              notch_q=args.notch_q)
        f = np.fft.rfftfreq(X_semg.shape[1], d=1 / 1000.0)
        def band_ratio(A, lo, hi):
            P = (np.abs(np.fft.rfft(A - A.mean(1, keepdims=True), axis=1)) ** 2).mean(axis=(0, 2))
            return P[(f >= lo) & (f < hi)].sum() / P[1:].sum() * 100
        for lo, hi, nm in [(1, 20, "1-20Hz"), (47.7, 48.7, "48Hz"), (51, 450, "51-450Hz")]:
            print(f"        {nm:10s} 滤波前 {band_ratio(before, lo, hi):6.2f}% -> "
                  f"滤波后 {band_ratio(X_semg, lo, hi):6.2f}%")
    else:
        print("\n[1.5/5] sEMG 不滤波（--filter none）")

    # 2. 划分
    print("\n[2/5] 分层划分 8:1:1 (以 .pkl 为单位) ...")
    split = strat_split(y, args.ratio, args.seed)
    tr = split == "train"; va = split == "val"; te = split == "test"
    print(f"  训练 {tr.sum()} | 验证 {va.sum()} | 测试 {te.sum()}")

    # 3. 训练集统计
    print("\n[3/5] 计算训练集归一化统计量 ...")
    st_g = compute_norm_stats(X_glove[tr], args.clip_q, args.dead_std_thr)
    st_s = compute_norm_stats(X_semg[tr], args.clip_q, args.dead_std_thr)
    print(f"  glove 死通道 {int(st_g['dead'].sum())}/40 : {np.where(st_g['dead'])[0].tolist()}"
          f" | 近恒定(std<0.01) {int(st_g['near_const'].sum())}/40 : {np.where(st_g['near_const'])[0].tolist()}")
    print(f"  semg  死通道 {int(st_s['dead'].sum())}/20 : {np.where(st_s['dead'])[0].tolist()}"
          f" | 近恒定(std<0.01) {int(st_s['near_const'].sum())}/20 : {np.where(st_s['near_const'])[0].tolist()}")

    # 4. 应用
    print(f"\n[4/5] 应用截断+归一化 (mode={args.norm_mode}, clip={not args.no_clip}) ...")
    Xg = apply_norm(X_glove, st_g, clip=not args.no_clip, mode=args.norm_mode)
    Xs = apply_norm(X_semg, st_s, clip=not args.no_clip, mode=args.norm_mode)
    print(f"  归一化后 glove: mean={Xg.mean():.4f} std={Xg.std():.4f} "
          f"min={Xg.min():.3f} max={Xg.max():.3f}")
    print(f"  归一化后 semg : mean={Xs.mean():.4f} std={Xs.std():.4f} "
          f"min={Xs.min():.3f} max={Xs.max():.3f}")

    # 5. 保存
    print("\n[5/5] 保存缓存 ...")
    C = os.path.join(args.cache_dir, args.out_subdir) if args.out_subdir else args.cache_dir
    os.makedirs(C, exist_ok=True)
    np.save(os.path.join(C, "X_glove.npy"), Xg)
    np.save(os.path.join(C, "X_semg.npy"), Xs)
    np.save(os.path.join(C, "y.npy"), y)
    if not args.no_save_raw:
        np.save(os.path.join(C, "X_glove_raw.npy"), X_glove)
        np.save(os.path.join(C, "X_semg_raw.npy"), X_semg)
    np.savez(os.path.join(C, "norm_stats.npz"),
             glove_mean=st_g["mean"], glove_std=st_g["std"], glove_lo=st_g["lo"],
             glove_hi=st_g["hi"], glove_dead=st_g["dead"],
             semg_mean=st_s["mean"], semg_std=st_s["std"], semg_lo=st_s["lo"],
             semg_hi=st_s["hi"], semg_dead=st_s["dead"])
    with open(os.path.join(C, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({c: i for i, c in enumerate(classes)}, f, ensure_ascii=False, indent=2)
    import pandas as pd
    pd.DataFrame({
        "index": np.arange(len(y)), "label": y,
        "class": [m[1] for m in meta], "file": [m[2] for m in meta], "split": split,
    }).to_csv(os.path.join(C, "meta.csv"), index=False, encoding="utf-8-sig")

    report = {
        "data_root": args.data_root, "cache_dir": C,
        "num_classes": len(classes), "num_samples": int(len(y)),
        "classes": classes,
        "shapes": {"glove": list(Xg.shape), "semg": list(Xs.shape)},
        "split": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum()),
                  "ratio": list(args.ratio), "seed": args.seed},
        "norm": {"mode": args.norm_mode, "clip": not args.no_clip,
                 "clip_quantile": list(args.clip_q), "dead_std_thr": args.dead_std_thr,
                 "glove_dead_ch": np.where(st_g["dead"])[0].tolist(),
                 "semg_dead_ch": np.where(st_s["dead"])[0].tolist(),
                 "glove_near_const_ch": np.where(st_g["near_const"])[0].tolist(),
                 "semg_near_const_ch": np.where(st_s["near_const"])[0].tolist()},
        "filter": {"mode": args.filter, "bandpass": list(args.bp),
                   "bp_order": args.bp_order, "notch_freqs": list(args.notch_freqs),
                   "notch_q": args.notch_q},
    }
    with open(os.path.join(C, "preprocess_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n完成。产物目录:", C)
    for fn in sorted(os.listdir(C)):
        p = os.path.join(C, fn)
        print(f"  {fn:26s} {os.path.getsize(p) / 1e6:8.2f} MB")


if __name__ == "__main__":
    main()
