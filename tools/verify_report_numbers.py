# -*- coding: utf-8 -*-
"""报告数值一致性核查：把 main.tex 中手写的关键数字与源产物逐一比对。"""
import os
import re
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "outputs", "results")
CACHE = os.path.join(ROOT, "outputs", "cache")
tex = open(os.path.join(ROOT, "reports", "main.tex"), encoding="utf-8").read()

checks = []


def ck(name, expect_str, actual, tol=5e-5):
    """expect_str: 报告里写的百分数字符串，如 '98.61'；actual: 真实比例 0~1"""
    ok = abs(float(expect_str) / 100 - actual) < tol
    checks.append((ok, name, expect_str, f"{actual*100:.4f}"))


def has(s, name):
    checks.append((s in tex, f"[文本存在] {name}", "—", "—"))


def load_summary(tag):
    return json.load(open(os.path.join(RES, f"{tag}_summary.json"), encoding="utf-8"))


# --- 主模型（G0：bilstm_both，100 轮预算，报告第 6 节与第 11 节引用）---
s = load_summary("bilstm_both")
ck("主模型 test_acc = 98.61%", "98.61", s["test_accuracy"])
ck("主模型 precision = 98.66%", "98.66", s["test_precision_macro"])
ck("主模型 best_val = 98.96%", "98.96", s["best_val_acc"])
checks.append((s["best_epoch"] == 35, "主模型 best_epoch = 35", str(s["best_epoch"]), "35"))
checks.append((abs(s["train_time_sec"] - 1378) < 1, "主模型用时 1378s",
               f"{s['train_time_sec']:.0f}", "1378"))
checks.append((s["n_params"] == 16151320, "参数量 16.15M",
               f"{s['n_params']/1e6:.2f}M", "16.15M"))

# --- 60 轮预算下的 Bi-LSTM（模型对比 / 消融基准引用）---
s60 = load_summary("abl_A0_full")
ck("60轮基准 test_acc = 98.96%", "98.96", s60["test_accuracy"])
ck("60轮基准 best_val = 99.31%", "99.31", s60["best_val_acc"])
checks.append((s60["best_epoch"] == 29, "60轮基准 best_epoch = 29",
               str(s60["best_epoch"]), "29"))

# --- 第 6 节曲线描述 ---
h = pd.read_csv(os.path.join(RES, "bilstm_both_history.csv"))
checks.append((len(h) == 55, "主模型实际训练 55 轮", str(len(h)), "55"))
checks.append((abs(h.train_loss.iloc[0] - 3.0785) < 1e-3, "ep1 train_loss = 3.08",
               f"{h.train_loss.iloc[0]:.4f}", "3.0785"))
checks.append((abs(h.train_loss.iloc[-1] - 0.019) < 1e-3, "末轮 train_loss ≈ 0.019",
               f"{h.train_loss.iloc[-1]:.4f}", "0.0187"))
checks.append((abs(h.train_acc.iloc[-1] - 0.9961) < 5e-4, "末轮 train_acc = 99.61%",
               f"{h.train_acc.iloc[-1]*100:.2f}", "99.61"))

# --- 参数组合验证（报告表 8 / 第 7.2 节）---
bc = load_summary("best_combo")
ck("组合验证 test_acc = 98.26%", "98.26", bc["test_accuracy"])
ck("组合验证 best_val = 97.92%", "97.92", bc["best_val_acc"])
checks.append((bc["best_epoch"] == 22, "组合验证 best_epoch = 22",
               str(bc["best_epoch"]), "22"))
hb = pd.read_csv(os.path.join(RES, "best_combo_history.csv"))
checks.append((abs(hb.train_acc.iloc[-1] - 1.0) < 1e-6, "组合末轮 train_acc = 100.00%",
               f"{hb.train_acc.iloc[-1]*100:.2f}", "100.00"))
checks.append((abs(hb.val_acc.iloc[-1] - 0.9688) < 5e-4, "组合末轮 val_acc = 96.88%",
               f"{hb.val_acc.iloc[-1]*100:.2f}", "96.88"))
bs32 = float(pd.read_csv(os.path.join(RES, "param_sweep_all.csv"))
             .loc[lambda d: d.tag == "sweep_batch_size=32", "test_accuracy"].iloc[0])
checks.append((bc["test_accuracy"] < bs32,
               "组合 98.26% < 单项最优 99.31%（叠加不成立）",
               f"{bc['test_accuracy']*100:.2f}", f"{bs32*100:.2f}"))

# --- 多模态 ---
for tag, pct in [("modality_glove", "98.26"), ("modality_semg", "91.32"),
                 ("modality_both", "98.61")]:
    ck(f"{tag} test_acc = {pct}%", pct, load_summary(tag)["test_accuracy"])

# --- 模型对比 ---
for tag, pct in [("model_tcn", "99.65"), ("model_cnn", "99.31"),
                 ("model_bilstm", "98.96"), ("model_gru", "98.61")]:
    ck(f"{tag} test_acc = {pct}%", pct, load_summary(tag)["test_accuracy"])

# --- 消融 ---
for tag, pct in [("abl_A0_full", "98.96"), ("abl_A1_no_fc", "96.53"),
                 ("abl_A2_uni_lstm", "58.68"), ("abl_A3_layers1", "95.14"),
                 ("abl_A4_fusion_sum", "97.92")]:
    ck(f"{tag} test_acc = {pct}%", pct, load_summary(tag)["test_accuracy"])

d = pd.read_csv(os.path.join(RES, "ablation.csv"))
base = float(d.loc[d.tag == "abl_A0_full", "test_accuracy"].iloc[0])
for tag, dd in [("abl_A1_no_fc", "-2.43"), ("abl_A3_layers1", "-3.82"), ("abl_A4_fusion_sum", "-1.04")]:
    v = (float(d.loc[d.tag == tag, "test_accuracy"].iloc[0]) - base) * 100
    checks.append((abs(v - float(dd)) < 0.005, f"消融 Δ{tag} = {dd}pp", f"{v:+.2f}", dd))

# --- 参数扫描 ---
sw = pd.read_csv(os.path.join(RES, "param_sweep_all.csv"))


def get(tag):
    return float(sw.loc[sw.tag == tag, "test_accuracy"].iloc[0])


for tag, pct in [("sweep_lr=0.0005", "97.92"), ("sweep_lr=0.001", "98.61"),
                 ("sweep_lr=0.002", "98.96"), ("sweep_lr=0.005", "98.61"),
                 ("sweep_batch_size=16", "98.61"), ("sweep_batch_size=32", "99.31"),
                 ("sweep_batch_size=50", "98.61"),
                 ("sweep_dropout=0.1", "99.31"), ("sweep_dropout=0.3", "98.61"),
                 ("sweep_dropout=0.5", "97.57")]:
    ck(f"{tag} test_acc = {pct}%", pct, get(tag))

# dropout=0.5 的 val
dv = float(sw.loc[sw.tag == "sweep_dropout=0.5", "best_val_acc"].iloc[0])
checks.append((abs(dv - 0.9618) < 5e-5, "dropout0.5 val = 96.18%", f"{dv*100:.2f}", "96.18"))
lv = float(sw.loc[sw.tag == "sweep_lr=0.005", "best_val_acc"].iloc[0])
checks.append((abs(lv - 0.9653) < 5e-5, "lr0.005 val = 96.53%", f"{lv*100:.2f}", "96.53"))

# --- 近重复 ---
df = pd.read_csv(os.path.join(RES, "bilstm_both_test_pred.csv"))
y = np.load(os.path.join(CACHE, "y.npy"))
meta = pd.read_csv(os.path.join(CACHE, "meta.csv"))
tr, te = meta.split.values == "train", meta.split.values == "test"
Xg = np.load(os.path.join(CACHE, "X_glove_raw.npy")).reshape(2880, -1).astype(np.float64)
Z = Xg - Xg.mean(1, keepdims=True)
Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
mx = (Z[te] @ Z[tr].T).max(1)
ok = (df.y_true.values == df.y_pred.values)
dup = mx >= 0.999
checks.append((int(dup.sum()) == 79, "近重复样本数 = 79", str(int(dup.sum())), "79"))
checks.append((abs(dup.mean() * 100 - 27.4) < 0.05, "近重复占比 = 27.4%",
               f"{dup.mean()*100:.1f}", "27.4"))
checks.append((ok[dup].mean() == 1.0, "近重复子集准确率 = 100.00%",
               f"{ok[dup].mean()*100:.2f}", "100.00"))
checks.append((abs(ok[~dup].mean() * 100 - 98.09) < 0.005, "非近重复子集 = 98.09%",
               f"{ok[~dup].mean()*100:.2f}", "98.09"))
overall = ok.mean()
checks.append((abs((overall - ok[~dup].mean()) * 100 - 0.52) < 0.02,
               "近重复抬高量 = +0.52pp", f"{(overall-ok[~dup].mean())*100:+.2f}", "+0.52"))

# sEMG 近重复
Xs = np.load(os.path.join(CACHE, "X_semg_raw.npy")).reshape(2880, -1).astype(np.float64)
Zs = Xs - Xs.mean(1, keepdims=True)
Zs /= np.linalg.norm(Zs, axis=1, keepdims=True) + 1e-9
mxs = (Zs[te] @ Zs[tr].T).max(1)
dfs = pd.read_csv(os.path.join(RES, "modality_semg_test_pred.csv"))
oks = (dfs.y_true.values == dfs.y_pred.values)
dups = mxs >= 0.999
checks.append((abs(oks[~dups].mean() * 100 - 89.50) < 0.01, "sEMG 非近重复 = 89.50%",
               f"{oks[~dups].mean()*100:.2f}", "89.50"))
checks.append((abs((oks.mean() - oks[~dups].mean()) * 100 - 1.82) < 0.02,
               "sEMG 抬高量 = +1.82pp", f"{(oks.mean()-oks[~dups].mean())*100:+.2f}", "+1.82"))

# --- 文本存在性 ---
for key in ["98.61", "99.65", "83.75", "88.02", "16.15", "10.66", "5.46",
            "48.17", "45.9", "18.7", "11.2", "0.52", "1.82", "27.4"]:
    has(key, f"报告含 {key}")
has("0.002", "报告含扫描最优 lr 0.002")
has("不可简单叠加", "报告含组合验证结论")
has("99.31", "报告含最终采用配置的结果 99.31")

# --- 补充实验（第 15 节 / 表 16-20）---
RR = os.environ.get("SIGNLANG_ENVB_RESULTS", os.path.join(ROOT, "outputs", "env_b"))


def rload(tag):
    return json.load(open(os.path.join(RR, f"{tag}_summary.json"), encoding="utf-8"))


bilstm_seeds = ["remote_verify", "p3_bilstm_s0", "p3_bilstm_s1", "p3_bilstm_s2"]
tcn_seeds = ["p3_tcn_s0", "p3_tcn_s1", "p3_tcn_s2"]
b_acc = [rload(t)["test_accuracy"] * 100 for t in bilstm_seeds]
t_acc = [rload(t)["test_accuracy"] * 100 for t in tcn_seeds]
checks.append((abs(np.mean(b_acc) - 98.96) < 0.005, "Bi-LSTM 种子均值 = 98.96%",
               f"{np.mean(b_acc):.2f}", "98.96"))
checks.append((abs(np.std(b_acc, ddof=1) - 0.40) < 0.005, "Bi-LSTM 种子标准差 = 0.40pp",
               f"{np.std(b_acc,ddof=1):.2f}", "0.40"))
checks.append((abs(np.mean(t_acc) - 99.42) < 0.005, "TCN 种子均值 = 99.42%",
               f"{np.mean(t_acc):.2f}", "99.42"))
checks.append((abs(np.std(t_acc, ddof=1) - 0.40) < 0.005, "TCN 种子标准差 = 0.40pp",
               f"{np.std(t_acc,ddof=1):.2f}", "0.40"))
for tag, pct in [("p1_moddrop30", "98.96"), ("p1_moddrop50", "98.61"),
                 ("p2_base_semg", "89.24"), ("p2_filter_both", "95.49"),
                 ("p2_filter_semg", "75.69"), ("p3_bilstm_s0", "98.61"),
                 ("p3_bilstm_s1", "99.31"), ("p3_bilstm_s2", "98.61"),
                 ("p3_tcn_s0", "99.65"), ("p3_tcn_s1", "99.65"),
                 ("p3_tcn_s2", "98.96")]:
    ck(f"环境 B {tag} = {pct}%", pct, rload(tag)["test_accuracy"])

# 剪枝结果
pc = json.load(open(os.path.join(RR, "pruning_check.json"), encoding="utf-8"))
for tag, exp_g, exp_s, exp_same in [("remote_verify", 99.31, 4.17, 288),
                                    ("p1_moddrop30", 98.96, 5.90, 288),
                                    ("p1_moddrop50", 98.61, 4.17, 288),
                                    ("p2_filter_both", 59.03, 19.79, 168)]:
    d = pc[tag]
    checks.append((abs(d["glove_only"] * 100 - exp_g) < 0.005, f"剪枝 {tag} 只留手套 = {exp_g}%",
                   f"{d['glove_only']*100:.2f}", str(exp_g)))
    checks.append((abs(d["semg_only"] * 100 - exp_s) < 0.005, f"剪枝 {tag} 只留sEMG = {exp_s}%",
                   f"{d['semg_only']*100:.2f}", str(exp_s)))
    checks.append((d["identical_pred_full_vs_gloveonly"] == exp_same,
                   f"剪枝 {tag} 预测相同数 = {exp_same}",
                   str(d["identical_pred_full_vs_gloveonly"]), str(exp_same)))

# --- Oracle 上限分析（第 15.6 节 / 表 21）---
orc = json.load(open(os.path.join(RR, "oracle_analysis.json"), encoding="utf-8"))
orc_by = {r["model"]: r for r in orc}
for key, exp_base, exp_semg, exp_oracle in [
        ("local_bilstm", 98.61, 91.32, 98.96),
        ("local_tcn", 99.65, 91.32, 99.65),
        ("local_cnn", 99.31, 91.32, 99.31),
        ("remote_bilstm", 99.31, 89.24, 99.31),
        ("remote_tcn", 99.65, 89.24, 99.65)]:
    r = orc_by[key]
    checks.append((abs(r["baseline"] * 100 - exp_base) < 0.005, f"Oracle {key} 基线 = {exp_base}%",
                   f"{r['baseline']*100:.2f}", str(exp_base)))
    checks.append((abs(r["semg_only"] * 100 - exp_semg) < 0.005, f"Oracle {key} 仅sEMG = {exp_semg}%",
                   f"{r['semg_only']*100:.2f}", str(exp_semg)))
    checks.append((abs(r["oracle"] * 100 - exp_oracle) < 0.005, f"Oracle {key} 上限 = {exp_oracle}%",
                   f"{r['oracle']*100:.2f}", str(exp_oracle)))
checks.append((orc_by["remote_bilstm"]["both_wrong"] == 2, "环境 B 基线两者都错 = 2 个",
               str(orc_by["remote_bilstm"]["both_wrong"]), "2"))
checks.append((orc_by["remote_bilstm"]["semg_hurts"] == 29, "sEMG 拖后腿样本 = 29 个",
               str(orc_by["remote_bilstm"]["semg_hurts"]), "29"))

# --- 输出 ---
fail = [c for c in checks if not c[0]]
print(f"核查项 {len(checks)} 个，通过 {len(checks)-len(fail)}，失败 {len(fail)}")
print("=" * 78)
for ok_, name, exp, act in checks:
    if not ok_:
        print(f"  ❌ {name}: 报告={exp}  实际={act}")
if not fail:
    print("  ✅ 全部通过 —— 报告中的数字与实验产物完全一致")
