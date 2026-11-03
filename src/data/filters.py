# -*- coding: utf-8 -*-
"""
sEMG 滤波工具（阶段一预处理可选步骤）

依据实测诊断（tools/diag_semg_spectrum.py）：
  - 48.17 Hz 强单音干扰（-3dB 带宽仅 0.50Hz，占平均能量 18.7%），96.3/144.5/192.7/240.9 Hz 有谐波
  - 1–20 Hz 低频漂移/运动伪迹占 45.9%
  - 典型 sEMG 带 51–450 Hz 仅占 11.2%
→ 推荐：4 阶 Butterworth 带通 20–450 Hz（零相位 filtfilt）+ 48.17 Hz 及其谐波陷波。
  注意主峰在 48.17 Hz 而非 50 Hz，按 50 Hz 设计陷波会漏掉干扰。
"""
import os

import numpy as np
from scipy import signal


def design_bandpass(fs, lo=20.0, hi=450.0, order=4):
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.99)
    return signal.butter(order, [lo / nyq, hi / nyq], btype="bandpass")


def design_notch(fs, freq, q=30.0):
    b, a = signal.iirnotch(freq / (fs / 2.0), q)
    return b, a


def apply_bandpass(x, fs, lo=20.0, hi=450.0, order=4):
    """x: (N,T,C) -> 带通滤波（零相位）。"""
    b, a = design_bandpass(fs, lo, hi, order)
    return signal.filtfilt(b, a, x, axis=1).astype(np.float32)


def apply_notch(x, fs, freqs=(48.17, 96.34, 144.51), q=30.0):
    """x: (N,T,C) -> 逐谐波陷波（零相位）。"""
    out = x.astype(np.float64, copy=True)
    for f in freqs:
        if f >= fs / 2 * 0.98:
            continue
        b, a = design_notch(fs, f, q)
        out = signal.filtfilt(b, a, out, axis=1)
    return out.astype(np.float32)


def apply_filter(x, fs, mode="none", bp=(20.0, 450.0), order=4,
                 notch_freqs=(48.17, 96.34, 144.51), notch_q=30.0):
    """mode ∈ {none, bandpass, notch, bandpass+notch}。x: (N,T,C)"""
    if mode == "none":
        return x.astype(np.float32)
    out = x
    if "bandpass" in mode:
        out = apply_bandpass(out, fs, bp[0], bp[1], order)
    if "notch" in mode:
        out = apply_notch(out, fs, notch_freqs, notch_q)
    return out.astype(np.float32)


if __name__ == "__main__":
    import sys
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, _ROOT)
    fs = 1000.0
    X = np.load(os.path.join(_ROOT, "outputs", "cache", "X_semg_raw.npy"))
    X = X[:200]
    f = np.fft.rfftfreq(X.shape[1], d=1 / fs)
    tot = lambda A: (np.abs(np.fft.rfft(A - A.mean(1, keepdims=True), axis=1)) ** 2).mean(axis=(0, 2))
    for mode in ["none", "bandpass", "notch", "bandpass+notch"]:
        Y = apply_filter(X, fs, mode)
        P = tot(Y)
        band = lambda lo, hi: P[(f >= lo) & (f < hi)].sum() / P[1:].sum() * 100
        print(f"{mode:16s} 1-20Hz={band(1,20):5.2f}%  48Hz±0.5={band(47.7,48.7):5.2f}%  "
              f"51-450Hz={band(51,450):5.2f}%")
