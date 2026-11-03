# -*- coding: utf-8 -*-
"""实验脚本共享工具：批量训练、结果聚合、绘图。"""
import os
import copy
import json
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.train import train_one, load_config      # noqa: E402


def default_cfg():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return load_config(os.path.join(root, "configs", "config.yaml"))


def run_grid(cfg, overrides_list, label="exp"):
    """依次训练一组配置，返回 summary 列表。"""
    results = []
    for i, ov in enumerate(overrides_list, 1):
        print(f"\n===== [{label}] {i}/{len(overrides_list)}: {ov.get('tag')} =====")
        c = copy.deepcopy(cfg)
        # 允许通过 ov['cfg_patch'] 深度修改配置（如 num_layers）
        for k, v in ov.pop("cfg_patch", {}).items():
            node = c
            keys = k.split(".")
            for kk in keys[:-1]:
                node = node[kk]
            node[keys[-1]] = v
        results.append(train_one(c, ov))
    return results


def save_results(results, out_csv):
    df = pd.DataFrame([{**{k: v for k, v in r.items() if k != "config"},
                        **{f"cfg_{k}": v for k, v in r.get("config", {}).items()}}
                       for r in results])
    cols = ["tag", "model", "modality", "n_params", "best_epoch", "best_val_acc",
            "test_accuracy", "test_precision_macro", "test_f1_macro",
            "cfg_batch_size", "cfg_epochs", "cfg_lr", "cfg_dropout"]
    cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]
    df = df[cols]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("\n结果已保存:", out_csv)
    print(df.to_string(index=False))
    return df


def plot_metric_bar(df, x_col, y_col, out_path, title="", hue=None, ylabel="Accuracy"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.9), 4.5))
    xs = np.arange(len(df))
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(df)))) if hue is None else None
    bars = ax.bar(xs, df[y_col].values, color=colors)
    for b, v in zip(bars, df[y_col].values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(df[x_col].values, rotation=30, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title, pad=12)
    ax.set_ylim(0, min(1.15, max(df[y_col].max() * 1.16, 0.1)))   # 留顶部余量给数值标签
    ax.grid(alpha=.25, axis="y")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print("图已保存:", out_path)


def plot_param_line(df, param_col, out_path, ylabel="Accuracy"):
    """参数扫描折线图（x=参数值，y=val/test 准确率）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    d = df.sort_values(param_col)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(d[param_col].astype(str), d["best_val_acc"], "o-", label="Val Acc")
    ax.plot(d[param_col].astype(str), d["test_accuracy"], "s--", label="Test Acc")
    best = d.loc[d["best_val_acc"].idxmax()]
    ax.axhline(best["best_val_acc"], color="green", ls=":", alpha=.6)
    ax.annotate(f"best: {best[param_col]}", xy=(list(d[param_col].astype(str)).index(str(best[param_col])),
                                                best["best_val_acc"]),
                xytext=(0, 8), textcoords="offset points", ha="center", color="green", fontsize=9)
    ax.set_xlabel(param_col); ax.set_ylabel(ylabel)
    ax.set_title(f"参数调节: {param_col}")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print("图已保存:", out_path)
