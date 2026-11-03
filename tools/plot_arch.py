# -*- coding: utf-8 -*-
"""生成主模型（双分支 Bi-LSTM 融合）结构示意图，用于报告第 3 项。
输出: outputs/figures/model_arch.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "figures", "model_arch.png")

C_IN = "#2a9d8f"; C_LSTM = "#1f4e79"; C_FC = "#e76f51"; C_FUSE = "#6a4c93"; C_OUT = "#264653"


def box(ax, x, y, w, h, text, color, fs=9, tc="white"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=color, ec="white", lw=1.2))
    ax.text(x, y, text, ha="center", va="center", color=tc, fontsize=fs, linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, color="#555"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.3, color=color))


fig, ax = plt.subplots(figsize=(14, 7.2))
ax.set_xlim(0, 14); ax.set_ylim(0, 7.2); ax.axis("off")

Y_G, Y_S = 5.4, 1.9          # 两条支路高度
BW, BH = 1.5, 0.85

# 输入
box(ax, 0.85, Y_G, 1.5, 0.95, "应变/压力输入\n(B, 400, 40)\n20+20 通道", C_IN, 8.5)
box(ax, 0.85, Y_S, 1.5, 0.95, "sEMG 输入\n(B, 1950, 20)\n10+10 通道", C_IN, 8.5)

# 归一化
box(ax, 2.6, Y_G, 0.95, 0.7, "归一化", "#8ab17d", 8.5)
box(ax, 2.6, Y_S, 0.95, 0.7, "归一化", "#8ab17d", 8.5)

# Bi-LSTM ×3
box(ax, 4.35, Y_G, 1.9, 1.1, "3 层堆叠 Bi-LSTM\nhidden = 128\n(双向, 2×128)", C_LSTM, 8.5)
box(ax, 4.35, Y_S, 1.9, 1.1, "3 层堆叠 Bi-LSTM\nhidden = 512\n(双向, 2×512)", C_LSTM, 8.5)

# FC ×2
box(ax, 6.35, Y_G, 1.5, 0.85, "FC 256 → 128\nReLU + Dropout", C_FC, 8)
box(ax, 6.35, Y_S, 1.5, 0.85, "FC 256 → 128\nReLU + Dropout", C_FC, 8)

# 特征融合
box(ax, 8.4, 3.65, 1.5, 0.95, "特征融合\nConcat (256)", C_FUSE, 9)

# 分类头
box(ax, 10.4, 3.65, 1.35, 0.9, "Dropout 0.3\n+ FC 24", C_OUT, 8.5)
box(ax, 12.4, 3.65, 1.35, 0.9, "输出\n24 类 logits", C_OUT, 8.5)

# 箭头
for y in (Y_G, Y_S):
    arrow(ax, 1.6, y, 2.12, y); arrow(ax, 3.08, y, 3.4, y)
    arrow(ax, 5.3, y, 5.6, y);  arrow(ax, 7.1, y, 7.65, y)
arrow(ax, 7.65, Y_G, 8.4, 4.12)
arrow(ax, 7.65, Y_S, 8.4, 3.18)
arrow(ax, 9.15, 3.65, 9.72, 3.65)
arrow(ax, 11.08, 3.65, 11.72, 3.65)

ax.text(7.0, 6.75, "主模型：双分支 Bi-LSTM 特征融合（对齐源论文 Figure 7c）",
        ha="center", fontsize=13, fontweight="bold")
ax.text(7.0, 0.55, "总参数量 16.15 M ｜ AdamW (lr=1e-3, wd=5e-3) ｜ batch=50 ｜ dropout=0.3 ｜ "
        "CosineAnnealing ｜ AMP(fp16) ｜ 早停 patience=20",
        ha="center", fontsize=9, color="#444")
ax.text(0.85, Y_G + 0.75, "手套支路（低频时序）", ha="center", fontsize=9, color=C_LSTM)
ax.text(0.85, Y_S + 0.75, "肌电支路（细粒度时序）", ha="center", fontsize=9, color=C_LSTM)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.tight_layout()
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print("saved", OUT)
