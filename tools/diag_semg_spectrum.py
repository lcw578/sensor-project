# -*- coding: utf-8 -*-
"""只读诊断：sEMG 频谱与方差结构，用于决定是否滤波 / 是否用 per-sample 标准化。
输出图: outputs/figures/diag_semg_spectrum.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "outputs", "cache")
FIG = os.path.join(ROOT, "outputs", "figures", "diag_semg_spectrum.png")

Xs = np.load(f"{CACHE}/X_semg_raw.npy")
Xg = np.load(f"{CACHE}/X_glove_raw.npy")
y = np.load(f"{CACHE}/y.npy")
N, T, C = Xs.shape
fs = 1000.0

# ---- 平均功率谱（原生分辨率）----
seg = Xs - Xs.mean(axis=1, keepdims=True)
f = np.fft.rfftfreq(T, d=1 / fs)
Pm = (np.abs(np.fft.rfft(seg, axis=1)) ** 2).mean(axis=(0, 2))
Pm_db = 10 * np.log10(Pm / Pm[1:].max() + 1e-12)
tot = Pm[1:].sum()

# ---- 逐样本 48.2Hz 干扰占比 ----
m48 = (f >= 47.7) & (f <= 48.7)
spike = (np.abs(np.fft.rfft(seg, axis=1)) ** 2)[:, m48, :].sum(axis=(1, 2)) / \
        (np.abs(np.fft.rfft(seg, axis=1)) ** 2)[:, 1:, :].sum(axis=(1, 2)) * 100

fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

# (a) 平均功率谱 + 频带标注
ax = fig.add_subplot(gs[0, :])
ax.plot(f, Pm_db, lw=0.9, color="#1f4e79")
ax.axvspan(0, 20, color="#f4a261", alpha=.25)
ax.axvspan(20, 450, color="#2a9d8f", alpha=.15)
ax.axvline(48.17, color="crimson", ls="--", lw=1.4)
ax.annotate("48.17 Hz 工频干扰\n(占总能量 18.7%)", xy=(48.17, Pm_db[int(48.17 / (fs / T))] + 1),
            xytext=(135, -20), color="crimson", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="crimson"))
ax.text(70, -52, "1–20 Hz 漂移/运动伪迹  占 45.9%（非 sEMG）",
        color="#b5651d", fontsize=9, ha="left")
ax.text(235, Pm_db.max() - 6, "典型 sEMG 带 20–450 Hz（仅占 11.2%）",
        color="#1b6b5a", fontsize=9, ha="center")
ax.set_xlim(0, 500); ax.set_ylim(-75, 5)
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Power (dB)")
ax.set_title("(a) sEMG 平均功率谱 —— 能量高度集中在低频与 48.17 Hz 单音", fontsize=11)
ax.grid(alpha=.25)

# (b) 逐样本干扰占比直方图
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(spike, bins=40, color="#c1121f", alpha=.8, edgecolor="white")
ax2.axvline(np.median(spike), color="black", ls="--", lw=1.4,
            label=f"中位 {np.median(spike):.1f}%")
ax2.axvline(np.percentile(spike, 90), color="#457b9d", ls=":", lw=1.6,
            label=f"P90 {np.percentile(spike,90):.1f}%")
ax2.axvline(30, color="green", ls="-.", lw=1.2, label="30% 阈值")
ax2.set_xlabel("单样本中 48.17 Hz 干扰占总能量比例 (%)")
ax2.set_ylabel("样本数")
ax2.set_title(f"(b) 干扰逐样本分布：{(spike>30).mean()*100:.0f}% 的样本 >30%", fontsize=10)
ax2.legend(fontsize=8); ax2.grid(alpha=.25)

# (c) 方差结构 & 幅值判别力
ax3 = fig.add_subplot(gs[1, 1])
labels = ["sEMG", "glove"]
between_within = [0.114, 3.017]
amp_f = [2.88, 0.135]
x = np.arange(2); w = 0.36
ax3.bar(x - w/2, between_within, w, label="样本间DC/样本内动态", color="#e76f51")
ax3.bar(x + w/2, amp_f, w, label="幅值(RMS)类判别力 F", color="#264653")
for i, v in enumerate(between_within):
    ax3.text(i - w/2, v + .05, f"{v:.2f}", ha="center", fontsize=9)
for i, v in enumerate(amp_f):
    ax3.text(i + w/2, v + .05, f"{v:.2f}", ha="center", fontsize=9)
ax3.set_xticks(x); ax3.set_xticklabels(labels)
ax3.set_title("(c) per-sample 标准化利弊依据", fontsize=10)
ax3.legend(fontsize=8); ax3.grid(alpha=.25, axis="y")

fig.suptitle("sEMG 预处理决策诊断（只读分析，未改动流水线）", fontsize=12, y=0.98)
fig.savefig(FIG, dpi=150, bbox_inches="tight")
print("saved", FIG)

# 频带能量汇总
print("\n频带能量占比:")
for lo, hi, nm in [(1, 20, "1-20Hz 漂移"), (20, 50, "20-50Hz"), (50, 150, "50-150Hz"),
                   (150, 450, "150-450Hz"), (450, 500, "450-500Hz")]:
    mm = (f >= lo) & (f < hi)
    print(f"  {nm:14s} {Pm[mm].sum()/tot*100:6.2f}%")
