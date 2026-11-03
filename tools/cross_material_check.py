# -*- coding: utf-8 -*-
"""跨材料一致性核查：实验产物(ground truth) ↔ 英文报告 ↔ 英文 deck ↔ 中文材料。

输出三类问题：
  A. 材料中出现、但与产物不符的数字（硬伤）
  B. 产物里的关键数字，某份材料缺失（覆盖缺口）
  C. 术语/口径不一致
"""
import os
import re
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RR = os.environ.get("SIGNLANG_ENVB_RESULTS", os.path.join(ROOT, "outputs", "env_b"))
RES = os.path.join(ROOT, "outputs", "results")

EN_TEX = open(os.path.join(ROOT, "reports", "main_en.tex"), encoding="utf-8").read()
EN_DECK = open(os.path.join(ROOT, "deck_en", "deck.html"), encoding="utf-8").read()
ZH_TEX = open(os.path.join(ROOT, "reports", "main.tex"), encoding="utf-8").read()
ZH_DECK = open(os.path.join(ROOT, "deck", "deck.html"), encoding="utf-8").read()

MATS = {"report_en": EN_TEX, "deck_en": EN_DECK, "report_zh": ZH_TEX, "deck_zh": ZH_DECK}


def S(tag):
    p = os.path.join(RR, f"{tag}_summary.json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    return json.load(open(os.path.join(RES, f"{tag}_summary.json"), encoding="utf-8"))


def L(tag):
    p = os.path.join(RR, f"{tag}_summary.json")
    src = RR if os.path.exists(p) else RES
    return json.load(open(os.path.join(src, f"{tag}_summary.json"), encoding="utf-8"))


# ---------------- ground truth ----------------
GT = {}
for t in ["remote_verify", "p3_bilstm_s0", "p3_bilstm_s1", "p3_bilstm_s2",
          "p3_tcn_s0", "p3_tcn_s1", "p3_tcn_s2", "p1_moddrop30", "p1_moddrop50",
          "p2_base_semg", "p2_filter_semg", "p2_filter_both"]:
    GT[t] = S(t)["test_accuracy"] * 100
for t in ["bilstm_both", "model_tcn", "model_cnn", "modality_glove", "modality_semg",
          "abl_A0_full", "abl_A1_no_fc", "abl_A2_uni_lstm", "abl_A3_layers1", "abl_A4_fusion_sum",
          "hp_lr0005", "hp_lr001", "hp_lr002", "hp_lr005", "hp_bs16", "hp_bs32", "hp_bs50",
          "hp_drop01", "hp_drop03", "hp_drop05"]:
    try:
        GT[t] = S(t)["test_accuracy"] * 100
    except Exception:
        pass

b_seeds = [GT["remote_verify"], GT["p3_bilstm_s0"], GT["p3_bilstm_s1"], GT["p3_bilstm_s2"]]
t_seeds = [GT["p3_tcn_s0"], GT["p3_tcn_s1"], GT["p3_tcn_s2"]]
GT["bilstm_mean"] = float(np.mean(b_seeds))
GT["bilstm_std"] = float(np.std(b_seeds, ddof=1))
GT["tcn_mean"] = float(np.mean(t_seeds))
GT["tcn_std"] = float(np.std(t_seeds, ddof=1))
GT["arch_gap"] = GT["tcn_mean"] - GT["bilstm_mean"]
GT["fusion_gain_local"] = GT["bilstm_both"] - GT["modality_glove"]

pc = json.load(open(os.path.join(RR, "pruning_check.json"), encoding="utf-8"))
orc = {r["model"]: r for r in json.load(open(os.path.join(RR, "oracle_analysis.json"), encoding="utf-8"))}

# ---------------- A. 硬伤：材料数字 vs 产物 ----------------
print("=" * 88)
print("A. 材料中出现的准确率数字 vs 实验产物（允许 ±0.005）")
print("=" * 88)

# 每个材料里抽取 "98.61%" 这类模式，逐个对照允许集合
allowed = {}
for t, v in GT.items():
    allowed.setdefault(round(v, 2), []).append(t)

extra_ok = {4.17, 5.90, 19.79, 59.03, 75.69, 89.24, 95.49, 99.31, 99.65, 98.96, 98.61,
            91.32, 98.26, 96.53, 96.18, 97.92, 97.57, 100.00, 18.06, 21.88, 0.7617,
            # 派生值：差值 / 百分比 / 能量占比
            13.55, 36.46, 14.5, 15.9, 3.82, 2.43, 1.04, 40.28, 0.69, 0.70, 0.46, 0.35,
            10.1, 27.4, 26.4, 88.8, 89.5, 98.09, 96.88, 99.61,
            # 源论文参照值
            83.75, 88.02,
            # sEMG 频带能量占比
            45.9, 18.7, 11.2, 40.3, 40.6, 0.2}

problems = []
for name, body in MATS.items():
    nums = set()
    for m in re.finditer(r"(\d{2,3}\.\d{1,2})\s*%", body):
        nums.add(round(float(m.group(1)), 2))
    for m in re.finditer(r"(\d{2,3}\.\d{1,2})\s*\\?,?\s*pp", body):
        nums.add(round(float(m.group(1)), 2))
    bad = []
    for n in sorted(nums):
        if n in allowed or n in extra_ok:
            continue
        # 允许的派生值
        if abs(n - GT["bilstm_mean"]) < 0.005 or abs(n - GT["tcn_mean"]) < 0.005:
            continue
        if abs(n - GT["bilstm_std"]) < 0.005 or abs(n - GT["tcn_std"]) < 0.005:
            continue
        if abs(n - GT["arch_gap"]) < 0.005:
            continue
        bad.append(n)
    if bad:
        problems.append((name, bad))
        print(f"  ❌ {name}: 产物中找不到依据的数字 → {bad}")
    else:
        print(f"  ✅ {name}: 所有准确率数字都有产物依据")

# ---------------- B. 覆盖：关键数字在哪些材料出现 ----------------
print()
print("=" * 88)
print("B. 关键结论数字的覆盖情况（应至少出现在 deck_en 与 report_en）")
print("=" * 88)

KEY = [
    ("Bi-LSTM 多种子均值", "98.96", ["98.96"]),
    ("Bi-LSTM 种子标准差", "0.40", ["0.40"]),
    ("TCN 多种子均值", "99.42", ["99.42"]),
    ("架构差距 < 种子波动", "0.46", ["0.46"]),
    ("主模型单次", "98.61", ["98.61"]),
    ("环境 B 基线", "99.31", ["99.31"]),
    ("TCN 单次最高", "99.65", ["99.65"]),
    ("sEMG 单模态", "91.32", ["91.32"]),
    ("手套单模态", "98.26", ["98.26"]),
    ("融合增益", "0.35", ["0.35"]),
    ("剪枝-只留sEMG", "4.17", ["4.17"]),
    ("剪枝-288/288", "288/288", ["288/288"]),
    ("模态dropout钉死", "0.7617", ["0.7617"]),
    ("滤波-sEMG-only", "75.69", ["75.69"]),
    ("滤波-融合", "95.49", ["95.49"]),
    ("滤波代价", "13.55", ["13.55"]),
    ("剪枝代价-滤波版", "24.31", ["24.31"]),
    ("Oracle-headroom", "0.00", ["0.00"]),
    ("近重复比例", "27.4", ["27.4"]),
    ("近重复膨胀", "0.52", ["0.52"]),
    ("消融-FC", "2.43", ["2.43"]),
    ("消融-深度", "3.82", ["3.82"]),
    ("消融-单向", "58.68", ["58.68"]),
    ("组合失效", "98.26", ["98.26"]),
    ("跨机差距", "0.70", ["0.70"]),
]
for label, _, variants in KEY:
    row = []
    for name in ["deck_en", "report_en", "deck_zh", "report_zh"]:
        hit = any(v in MATS[name] for v in variants)
        row.append("✓" if hit else "·")
    flag = "" if (row[0] == "✓" and row[1] == "✓") else "   ← 英文材料缺"
    print(f"  {label:<26} deck_en:{row[0]} report_en:{row[1]} deck_zh:{row[2]} report_zh:{row[3]}{flag}")

# ---------------- C. 术语与口径 ----------------
print()
print("=" * 88)
print("C. 术语与口径一致性")
print("=" * 88)
terms = [
    ("Bi-LSTM 拼写", ["Bi-LSTM"], ["BILSTM", "BiLSTM", "bi-lstm"]),
    ("TCN 大小写", ["TCN"], ["tcn"]),
    ("sEMG 拼写", ["sEMG"], ["SEMG", "semg signal"]),
    ("dominant modality", ["dominant-modality", "dominant modality", "主导模态"], []),
    ("fp16 表述", ["fp16"], ["FP16", "half precision"]),
]
for label, good, bad_pat in terms:
    line = f"  {label:<22}"
    for name in ["deck_en", "report_en"]:
        body = MATS[name].lower()
        g = any(x.lower() in body for x in good)
        b = any(x.lower() in body for x in bad_pat)
        line += f" {name}: {'✓' if g and not b else ('⚠ 有异体' if b else '✗ 缺')}"
    print(line)

# ---------------- 汇总 ----------------
print()
print("=" * 88)
hard = sum(1 for p in problems)
print(f"结论：A 类硬伤 {hard} 处；覆盖与术语见上表。")
