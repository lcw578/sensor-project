# -*- coding: utf-8 -*-
"""deck 专用英文配图（剪枝 / 滤波 / Oracle）→ deck_en/media/"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RR = os.environ.get("SIGNLANG_ENVB_RESULTS", os.path.join(ROOT, "outputs", "env_b"))
OUT = os.path.join(ROOT, "deck_en", "media")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
ACC, RED, GREY, ORANGE = "#0071E3", "#D0342C", "#8E8E93", "#E76F51"


def save(fig, n):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, n), bbox_inches="tight")
    plt.close(fig)
    print("  ->", n)


def loader(tag):
    return json.load(open(os.path.join(RR, f"{tag}_summary.json"), encoding="utf-8"))


# ---------------------------------------------------------------- 剪枝
def fig_pruning():
    pc = json.load(open(os.path.join(RR, "pruning_check.json"), encoding="utf-8"))
    tags = ["remote_verify", "p1_moddrop30", "p1_moddrop50"]
    labels = ["Baseline\n(no dropout)", "Modality dropout\n30%", "Modality dropout\n50%"]
    full = [pc[t]["full"] * 100 for t in tags]
    glove = [pc[t]["glove_only"] * 100 for t in tags]
    semg = [pc[t]["semg_only"] * 100 for t in tags]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(3); w = 0.26
    b1 = ax.bar(x - w, full, w, label="Full fusion", color=ACC)
    b2 = ax.bar(x, glove, w, label="sEMG zeroed (glove only)", color="#5AC8FA")
    b3 = ax.bar(x + w, semg, w, label="Glove zeroed (sEMG only)", color=RED)
    for bs in (b1, b2, b3):
        for b in bs:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                    f"{b.get_height():.1f}", ha="center", fontsize=10)
    for i, t in enumerate(tags):
        same = pc[t]["identical_pred_full_vs_gloveonly"]
        ax.text(i, 106, f"{same}/288 identical\npredictions", ha="center",
                fontsize=10, color="#1D1D1F", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(0, 118)
    ax.set_title("Zeroing sEMG changes nothing; zeroing the glove collapses to chance",
                 fontsize=12, pad=26)
    ax.legend(fontsize=10, loc="center right", framealpha=.95); ax.grid(alpha=.25, axis="y")
    save(fig, "fig7_pruning.png")


# ---------------------------------------------------------------- 滤波
def fig_filter():
    fig, ax = plt.subplots(figsize=(9.5, 5))
    labels = ["sEMG only\n(unfiltered)", "sEMG only\n(filtered)", "Fused\n(unfiltered)", "Fused\n(filtered)"]
    vals = [loader("p2_base_semg")["test_accuracy"] * 100,
            loader("p2_filter_semg")["test_accuracy"] * 100,
            loader("remote_verify")["test_accuracy"] * 100,
            loader("p2_filter_both")["test_accuracy"] * 100]
    cols = [ACC, RED, ACC, ORANGE]
    bars = ax.bar(np.arange(4), vals, color=cols, width=.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.2f}", ha="center", fontsize=11)
    ax.annotate("", xy=(0, 92), xytext=(1, 78), arrowprops=dict(arrowstyle="->", color=RED, lw=2))
    ax.text(0.5, 96, "$-13.55$ pp", color=RED, fontsize=13, ha="center", fontweight="bold")
    ax.annotate("", xy=(2, 97), xytext=(3, 96.5), arrowprops=dict(arrowstyle="->", color=RED, lw=2))
    ax.text(2.5, 99.5, "$-3.82$ pp", color=RED, fontsize=13, ha="center", fontweight="bold")
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(60, 106)
    ax.set_title("Filtering out drift and the 48 Hz line makes everything worse", fontsize=12)
    ax.grid(alpha=.25, axis="y")
    save(fig, "fig8_filter.png")


# ---------------------------------------------------------------- Oracle
def fig_oracle():
    rows = json.load(open(os.path.join(RR, "oracle_analysis.json"), encoding="utf-8"))
    names = {"local_bilstm": "Bi-LSTM\n(host A)", "local_tcn": "TCN\n(host A)",
             "local_cnn": "1D-CNN\n(host A)", "remote_bilstm": "Bi-LSTM\n(host B)",
             "remote_tcn": "TCN\n(host B)"}
    obs = [r["baseline"] * 100 for r in rows]
    orc = [r["oracle"] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(rows)); w = 0.34
    b1 = ax.bar(x - w / 2, obs, w, label="Observed accuracy", color=ACC)
    b2 = ax.bar(x + w / 2, orc, w, label="Oracle upper bound (any fusion)", color="#C7C7CC")
    for i, r in enumerate(rows):
        if abs(r["oracle"] - r["baseline"]) < 1e-9:
            ax.text(i, obs[i] + 0.12, "no headroom", ha="center", fontsize=10,
                    color=RED, fontweight="bold")
        else:
            ax.text(i, obs[i] + 0.12, f"+{r['headroom']*100:.2f} pp (1 sample)", ha="center",
                    fontsize=9.5, color=RED)
    ax.set_xticks(x); ax.set_xticklabels([names[r["model"]] for r in rows], fontsize=10.5)
    ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(97.5, 100.6)
    ax.set_title("Even an oracle that knows which model is right cannot beat the observed score",
                 fontsize=12)
    ax.legend(fontsize=10); ax.grid(alpha=.25, axis="y")
    save(fig, "fig9_oracle.png")


if __name__ == "__main__":
    print("deck figures ...")
    fig_pruning(); fig_filter(); fig_oracle()
    print("done ->", OUT)
