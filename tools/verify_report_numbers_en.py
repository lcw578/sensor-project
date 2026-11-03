# -*- coding: utf-8 -*-
"""核对英文报告中的关键数字与实验产物一致（防止翻译/重写时抄错）。"""
import os
import re
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EN = os.path.join(ROOT, "reports", "main_en.tex")
ZH = os.path.join(ROOT, "reports", "main.tex")
txt = open(EN, encoding="utf-8").read()
zh = open(ZH, encoding="utf-8").read()
RR = os.environ.get("SIGNLANG_ENVB_RESULTS", os.path.join(ROOT, "outputs", "env_b"))


def S(tag):
    return json.load(open(os.path.join(RR, f"{tag}_summary.json"), encoding="utf-8"))


checks = []


def has(s, name, src="en"):
    checks.append((s in (txt if src == "en" else zh), f"[{src}] contains: {name}", s, "-"))


def num(expect, actual, name, tol=5e-5):
    checks.append((abs(float(expect) / 100 - actual) < tol, name, expect, f"{actual*100:.4f}"))


# 关键数字必须出现在英文报告中
for s in ["2{,}880", "2304" if False else "2{,}304", "288", "96 / 12 / 12",
          "16.15", "98.61", "98.96", "99.42", "99.31", "91.32", "98.26",
          "0.40", "0.46", "13.55", "75.69", "89.24", "95.49", "59.03",
          "288/288", "4.17", "99.65", "58.68", "2.43", "3.82", "1.04",
          "27.4", "0.52", "1.82", "48.17", "45.9", "11.2", "0.7617",
          "51c86320", "81{,}627", "10.66", "5.46", "0.70", "21.88", "18.06"]:
    has(s, s)

# 与产物对齐
num("98.96", np.mean([S(t)["test_accuracy"] for t in
                      ["remote_verify", "p3_bilstm_s0", "p3_bilstm_s1", "p3_bilstm_s2"]]),
    "bilstm seed mean = 98.96%")
num("99.42", np.mean([S(t)["test_accuracy"] for t in
                      ["p3_tcn_s0", "p3_tcn_s1", "p3_tcn_s2"]]), "tcn seed mean = 99.42%")
num("99.31", S("remote_verify")["test_accuracy"], "remote_verify = 99.31%")
num("89.24", S("p2_base_semg")["test_accuracy"], "p2_base_semg = 89.24%")
num("75.69", S("p2_filter_semg")["test_accuracy"], "p2_filter_semg = 75.69%")
num("95.49", S("p2_filter_both")["test_accuracy"], "p2_filter_both = 95.49%")
num("98.96", S("p1_moddrop30")["test_accuracy"], "p1_moddrop30 = 98.96%")
num("98.61", S("p1_moddrop50")["test_accuracy"], "p1_moddrop50 = 98.61%")

# 英文版与中文版的关键数字集合应当一致（抽查若干）
for s in ["98.61", "99.31", "0.40", "13.55", "288/288", "0.7617", "27.4"]:
    checks.append(((s in txt) == (s in zh), f"EN/ZH 一致性: {s}", s, "-"))

fail = [c for c in checks if not c[0]]
print(f"英文报告数值核查: {len(checks)} 项, 通过 {len(checks)-len(fail)}, 失败 {len(fail)}")
for ok, name, exp, act in checks:
    if not ok:
        print(f"  ❌ {name}: 期望={exp} 实际={act}")
if not fail:
    print("  ✅ 全部通过 —— 英文报告数字与实验产物、中文版均一致")
