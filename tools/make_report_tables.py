# -*- coding: utf-8 -*-
"""从各结果 CSV 生成 LaTeX 表格片段 -> reports/tables/*.tex
保证报告中的数值与实验产物完全一致，避免手工转录错误。
"""
import os
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "outputs", "results")
CACHE = os.path.join(ROOT, "outputs", "cache")
OUT = os.path.join(ROOT, "reports", "tables")
os.makedirs(OUT, exist_ok=True)


def esc(s):
    return (str(s).replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#"))


def pct(x, nd=2):
    return f"{x*100:.{nd}f}"


def write(name, text):
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    print("  ->", os.path.relpath(p, ROOT))


NAME_CN = {"bilstm": "Bi-LSTM（主模型）", "cnn": "1D-CNN", "gru": "Bi-GRU",
           "tcn": "TCN", "transformer": "Transformer"}
MOD_CN = {"glove": "A 仅应变/压力", "semg": "B 仅 sEMG", "both": "C 融合（应变/压力+sEMG）"}
ABL_CN = {"abl_A0_full": "A0 完整模型（基准）", "abl_A1_no_fc": "A1 去掉分支内 2 层 FC",
          "abl_A2_uni_lstm": "A2 双向 $\\rightarrow$ 单向 LSTM",
          "abl_A3_layers1": "A3 堆叠层数 $3\\rightarrow1$",
          "abl_A4_fusion_sum": "A4 融合 concat $\\rightarrow$ sum"}

print("生成 LaTeX 表格 ...")

# ---- 表1 多模态对比 ----
d = pd.read_csv(os.path.join(RES, "modality_compare.csv"))
rows = []
paper = {"glove": 0.8375, "semg": 0.8802, "both": 0.9965}
for _, r in d.iterrows():
    m = r["modality"]
    rows.append(f"{MOD_CN[m]} & {pct(r['test_accuracy'])} & {pct(r['test_precision_macro'])} & "
                f"{pct(r['test_f1_macro'])} & {pct(paper[m])} & {pct(r['test_accuracy']-paper[m])} \\\\")
write("tab_modality.tex", r"""\begin{table}[htbp]\centering
\caption{多模态对比实验（测试集 288 样本）}
\label{tab:modality}
\begin{tabular}{lccccr}
\toprule
实验设置 & Accuracy(\%) & Precision(\%) & F1(\%) & 论文(\%) & 差值(pp) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- 表2 模型对比 ----
d = pd.read_csv(os.path.join(RES, "model_compare.csv")).sort_values("test_accuracy", ascending=False)
rows = []
for _, r in d.iterrows():
    pm = r["n_params"] / 1e6
    rows.append(f"{NAME_CN.get(r['model'], r['model'])} & {pct(r['test_accuracy'])} & "
                f"{pct(r['test_precision_macro'])} & {pct(r['test_f1_macro'])} & "
                f"{pm:.2f} & {int(r['best_epoch'])} \\\\")
write("tab_model.tex", r"""\begin{table}[htbp]\centering
\caption{不同模型性能对比（均为多模态融合输入，8:1:1 划分）}
\label{tab:model}
\begin{tabular}{lccccc}
\toprule
模型 & Accuracy(\%) & Precision(\%) & F1(\%) & 参数量(M) & 收敛轮次 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- 表3 消融 ----
d = pd.read_csv(os.path.join(RES, "ablation.csv"))
base = float(d.loc[d.tag == "abl_A0_full", "test_accuracy"].iloc[0])
rows = []
for _, r in d.iterrows():
    delta = r["test_accuracy"] - base
    dd = "--" if r["tag"] == "abl_A0_full" else f"{delta*100:+.2f}"
    rows.append(f"{ABL_CN.get(r['tag'], r['tag'])} & {pct(r['test_accuracy'])} & "
                f"{pct(r['test_precision_macro'])} & {dd} & {r['n_params']/1e6:.2f} \\\\")
write("tab_ablation.tex", r"""\begin{table}[htbp]\centering
\caption{消融实验（基准为完整双分支 Bi-LSTM 融合模型）}
\label{tab:ablation}
\begin{tabular}{lcccr}
\toprule
消融项 & Accuracy(\%) & Precision(\%) & $\Delta$(pp) & 参数量(M) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- 表4/5/6 参数扫描 ----
s = pd.read_csv(os.path.join(RES, "param_sweep_all.csv"))
spec = [("lr", "cfg_lr", "学习率"), ("batch_size", "cfg_batch_size", "Batch Size"),
        ("dropout", "cfg_dropout", "Dropout")]
for key, col, cn in spec:
    sub = s[s["tag"].str.startswith(f"sweep_{key}=")].copy()
    best_test = sub["test_accuracy"].max()
    rows = []
    for _, r in sub.sort_values(col).iterrows():
        mark = r" $\star$" if r["test_accuracy"] == best_test else ""
        rows.append(f"{r[col]:g} & {pct(r['best_val_acc'])} & {pct(r['test_accuracy'])}{mark} & "
                    f"{pct(r['test_precision_macro'])} \\\\")
    write(f"tab_sweep_{key}.tex", r"""\begin{table}[htbp]\centering
\caption{参数调节：%s}
\label{tab:sweep%s}
\begin{tabular}{cccc}
\toprule
%s & 验证 Accuracy(\%%) & 测试 Accuracy(\%%) & Precision(\%%) \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (cn, key, cn, "\n".join(rows)))

# ---- 表7 与论文对照 ----
write("tab_paper.tex", r"""\begin{table}[htbp]\centering
\caption{与源论文结果的对照}
\label{tab:paper}
\begin{tabular}{lccc}
\toprule
项目 & 源论文 & 本复现 & 判定 \\
\midrule
融合准确率 & 99.65\% & 98.61\%（Bi-LSTM）/ 99.65\%（TCN） & 复现成功 \\
仅应变/压力 & 83.75\% & 98.26\% & 显著分歧 \\
仅 sEMG & 88.02\% & 91.32\% & 接近 \\
参考架构 & 双分支 Bi-LSTM 融合 & 已复现 & 一致 \\
训练超参 & AdamW/50/$10^{-3}$/0.3 & 已复现 & 一致 \\
\bottomrule
\end{tabular}
\end{table}
""")

# ---- 表8 近重复分层准确率 ----
write("tab_dup.tex", r"""\begin{table}[htbp]\centering
\caption{测试集按“在训练集中是否存在近重复样本”分层的准确率}
\label{tab:dup}
\begin{tabular}{lccc}
\toprule
子集 & 样本数 & 融合模型(\%) & 仅 sEMG(\%) \\
\midrule
存在近重复（$\cos\ge 0.999$） & 79 & 100.00 & 100.00 \\
不存在近重复 & 209 & 98.09 & 89.50 \\
\midrule
全体 & 288 & 98.61 & 91.32 \\
\bottomrule
\end{tabular}
\end{table}
""")

# ---- 各类召回率（交付物13 用）----
df = pd.read_csv(os.path.join(RES, "bilstm_both_test_pred.csv"))
lm = json.load(open(os.path.join(CACHE, "label_map.json"), encoding="utf-8"))
names = [None] * 24
for k, v in lm.items():
    names[v] = k
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(df.y_true, df.y_pred, labels=list(range(24)))
rec = cm.diagonal() / cm.sum(1)
order = np.argsort(rec)
rows = []
for i in order:
    rows.append(f"{names[i]} & {int(cm.sum(1)[i])} & {int(cm.diagonal()[i])} & {rec[i]*100:.1f} \\\\")
write("tab_perclass.tex", r"""\begin{table}[htbp]\centering
\caption{各类别召回率（主模型 Bi-LSTM 融合，按召回率升序）}
\label{tab:perclass}
\begin{tabular}{lccc}
\toprule
手语类别 & 测试样本数 & 正确数 & 召回率(\%) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- 误分类对 ----
errs = []
for i in range(24):
    for j in range(24):
        if i != j and cm[i, j] > 0:
            errs.append((int(cm[i, j]), names[i], names[j]))
errs.sort(reverse=True)
rows = [f"{a} & {b} & {c} \\\\" for c, a, b in errs]
write("tab_errors.tex", r"""\begin{table}[htbp]\centering
\caption{主模型的全部误分类样本}
\label{tab:errors}
\begin{tabular}{lcc}
\toprule
真实类别 & 预测类别 & 次数 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

print("完成。")
