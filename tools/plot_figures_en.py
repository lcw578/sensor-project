# -*- coding: utf-8 -*-
"""为英文版报告重新生成全套配图（英文标注）。输出到 reports/figs_en/。"""
import os
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "outputs", "results")
RR = os.environ.get("SIGNLANG_ENVB_RESULTS", os.path.join(ROOT, "outputs", "env_b"))
OUT = os.path.join(ROOT, "reports", "figs_en")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False,
                     "figure.dpi": 150, "savefig.dpi": 150})
ACC = "#0071E3"


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  ->", name)


# ---------------------------------------------------------------- F1 架构图
def fig_arch():
    C_IN, C_LSTM, C_FC, C_FUSE, C_OUT = "#2a9d8f", "#1f4e79", "#e76f51", "#6a4c93", "#264653"
    fig, ax = plt.subplots(figsize=(14, 7.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.2); ax.axis("off")

    def box(ax, x, y, w, h, text, color, fs=9, tc="white"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc=color, ec="white", lw=1.2))
        ax.text(x, y, text, ha="center", va="center", color=tc, fontsize=fs, linespacing=1.4)

    def arrow(ax, x1, y1, x2, y2, color="#555"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.3, color=color))

    YG, YS = 5.4, 1.9
    box(ax, 0.95, YG, 1.6, 0.95, "Strain/pressure\n(B, 400, 40)\n20+20 channels", C_IN, 8.5)
    box(ax, 0.95, YS, 1.6, 0.95, "sEMG\n(B, 1950, 20)\n10+10 channels", C_IN, 8.5)
    box(ax, 2.7, YG, 0.95, 0.7, "Normalise", "#8ab17d", 8.5)
    box(ax, 2.7, YS, 0.95, 0.7, "Normalise", "#8ab17d", 8.5)
    box(ax, 4.45, YG, 1.95, 1.1, "3-layer Bi-LSTM\nhidden = 128\n(bidirectional)", C_LSTM, 8.5)
    box(ax, 4.45, YS, 1.95, 1.1, "3-layer Bi-LSTM\nhidden = 512\n(bidirectional)", C_LSTM, 8.5)
    box(ax, 6.5, YG, 1.6, 0.85, "FC 256 -> 128\nReLU + Dropout", C_FC, 8)
    box(ax, 6.5, YS, 1.6, 0.85, "FC 256 -> 128\nReLU + Dropout", C_FC, 8)
    box(ax, 8.65, 3.65, 1.6, 0.95, "Concatenation\n(256-d)", C_FUSE, 9)
    box(ax, 10.7, 3.65, 1.4, 0.9, "Dropout 0.3\n+ FC 24", C_OUT, 8.5)
    box(ax, 12.7, 3.65, 1.3, 0.9, "24-class\nlogits", C_OUT, 8.5)

    for y in (YG, YS):
        arrow(ax, 1.75, y, 2.22, y); arrow(ax, 3.18, y, 3.47, y)
        arrow(ax, 5.43, y, 5.7, y);  arrow(ax, 7.3, y, 7.85, y)
    arrow(ax, 7.85, YG, 8.65, 4.12); arrow(ax, 7.85, YS, 8.65, 3.18)
    arrow(ax, 9.45, 3.65, 10.0, 3.65); arrow(ax, 11.4, 3.65, 12.05, 3.65)

    ax.text(7.0, 6.8, "Dual-branch Bi-LSTM with late feature fusion",
            ha="center", fontsize=13, fontweight="bold")
    ax.text(7.0, 0.5, "16.15 M parameters  |  AdamW (lr 1e-3, wd 5e-3)  |  batch 50  |  dropout 0.3  |  "
            "cosine schedule  |  fp16 AMP  |  early stop patience 20",
            ha="center", fontsize=9, color="#444")
    ax.text(0.95, YG + 0.78, "glove branch (low-frequency dynamics)", ha="center", fontsize=9, color=C_LSTM)
    ax.text(0.95, YS + 0.78, "sEMG branch (fine-grained muscle activity)", ha="center", fontsize=9, color=C_LSTM)
    save(fig, "fig1_architecture.png")


# ---------------------------------------------------------------- F2 训练曲线
def fig_curves():
    h = pd.read_csv(os.path.join(RES, "bilstm_both_history.csv"))
    ep = np.arange(1, len(h) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(ep, h.train_loss, label="Train", color=ACC)
    axes[0].plot(ep, h.val_loss, label="Validation", color="#e76f51")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].plot(ep, h.train_acc * 100, label="Train", color=ACC)
    axes[1].plot(ep, h.val_acc * 100, label="Validation", color="#e76f51")
    best = int(h.val_acc.idxmax()) + 1
    axes[1].axvline(best, ls=":", color="green", alpha=.7)
    axes[1].annotate(f"best epoch {best}", xy=(best, h.val_acc.max() * 100),
                     xytext=(-10, -30), textcoords="offset points", color="green", fontsize=9)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=.3)
    save(fig, "fig2_curves.png")


# ---------------------------------------------------------------- F3 混淆矩阵
def fig_cm():
    from sklearn.metrics import confusion_matrix
    df = pd.read_csv(os.path.join(RES, "bilstm_both_test_pred.csv"))
    lm = json.load(open(os.path.join(ROOT, "outputs/cache/label_map.json"), encoding="utf-8"))
    names = [None] * 24
    for k, v in lm.items():
        names[v] = k
    cm = confusion_matrix(df.y_true, df.y_pred, labels=list(range(24)))
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10.5, 9))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(24)); ax.set_yticks(range(24))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised), 4 errors / 288 samples", fontsize=11)
    for i in range(24):
        for j in range(24):
            if cm[i, j] and i != j:
                ax.text(j, i, cm[i, j], ha="center", va="center", color="#D0342C",
                        fontsize=8, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, "fig3_confusion.png")


# ---------------------------------------------------------------- F4/F5/F6 柱状图
def _bar(labels, values, title, name, color=None, ylim_pad=1.18):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    xs = np.arange(len(values))
    bars = ax.bar(xs, values, color=color or ACC, width=.55)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Test accuracy (%)"); ax.set_title(title, fontsize=11)
    ax.set_ylim(0, max(values) * ylim_pad)
    ax.grid(alpha=.25, axis="y")
    save(fig, name)


def fig_bars():
    d = pd.read_csv(os.path.join(RES, "modality_compare.csv"))
    m = {"glove": "Glove only", "semg": "sEMG only", "both": "Fused"}
    _bar([m[x] for x in d.modality], (d.test_accuracy * 100).tolist(),
         "Single-modality versus fused input", "fig4_modality.png")

    # 多种子柱状图（环境 B + 环境 A 单次）
    def acc(tag, base=RR):
        return json.load(open(os.path.join(base, f"{tag}_summary.json"),
                              encoding="utf-8"))["test_accuracy"] * 100
    bl = [acc(t) for t in ["remote_verify", "p3_bilstm_s0", "p3_bilstm_s1", "p3_bilstm_s2"]]
    tc = [acc(t) for t in ["p3_tcn_s0", "p3_tcn_s1", "p3_tcn_s2"]]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs = np.arange(4)
    tcn_slots = [np.nan] + tc          # TCN 没有 host B 的 seed-42 运行
    ax.bar(xs - 0.19, bl, width=.36,
           label=f"Bi-LSTM (mean {np.mean(bl):.2f}$\\pm${np.std(bl,ddof=1):.2f})", color=ACC)
    ax.bar(xs + 0.19, tcn_slots, width=.36,
           label=f"TCN (mean {np.mean(tc):.2f}$\\pm${np.std(tc,ddof=1):.2f})", color="#e76f51")
    ax.set_xticks(xs); ax.set_xticklabels(["seed 42", "seed 0", "seed 1", "seed 2"])
    ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(97.5, 100)
    ax.set_title("Multi-seed results: the gap is smaller than the seed spread", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=.25, axis="y")
    save(fig, "fig5_multiseed.png")

    a = pd.read_csv(os.path.join(RES, "ablation.csv"))
    cn = {"abl_A0_full": "A0 Full", "abl_A1_no_fc": "A1 no branch FC",
          "abl_A2_uni_lstm": "A2 unidirect.", "abl_A3_layers1": "A3 1 layer",
          "abl_A4_fusion_sum": "A4 sum fusion"}
    _bar([cn[t] for t in a.tag], (a.test_accuracy * 100).tolist(),
         "Ablation (test accuracy)", "fig6_ablation.png", color="#6a4c93")


if __name__ == "__main__":
    print("generating English figures ...")
    fig_arch(); fig_curves(); fig_cm(); fig_bars()
    print("done ->", OUT)
