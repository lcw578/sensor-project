# -*- coding: utf-8 -*-
"""把原「补充实验」章节拆解并入正式章节结构。"""
import re

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "main.tex")
src = open(P, encoding="utf-8").read()
block = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "reports", "_sec15_block.tex"), encoding="utf-8").read()

# ---- 切出各小节（含各自的 \subsection 头）----
def cut(start_kw, end_kw=None):
    i = block.index(start_kw)
    j = block.index(end_kw) if end_kw else len(block)
    return block[i:j].rstrip() + "\n"

sec_env = cut("\\subsection{实验环境与跨机器复现性}", "\\subsection{多种子重复实验}")
sec_seed = cut("\\subsection{多种子重复实验}", "\\subsection{模态 Dropout")
sec_drop = cut("\\subsection{模态 Dropout", "\\subsection{sEMG 滤波对比")
sec_filt = cut("\\subsection{sEMG 滤波对比", "\\subsection{融合的理论上限")
sec_orac = cut("\\subsection{融合的理论上限", "\\subsection{补充实验的综合结论}")
sec_sum = cut("\\subsection{补充实验的综合结论}")

# ---- 1) 从 main.tex 中删除原第 15 章 ----
i = src.index("%=====================================================================\n\\section{补充实验")
j = src.index("%=====================================================================\n\\section{结论}")
src = src[:i] + src[j:]

# ---- 2) 第 5 章（模型主要参数）末尾插入「实验环境」----
env_new = sec_env.replace(
    "\\subsection{实验环境与跨机器复现性}",
    "\\subsection{实验环境与跨机器一致性}"
).replace(
    "因此后续所有环境 B 实验均以环境 B 自身基线（99.31\\%）为对照，不跨环境比较。",
    "\\textbf{因此本报告对每一项配置均在同一主机内部进行横向对比，不跨机器比较绝对值}；"
    "涉及随机性的结论（如模型间差异）则以多种子统计为准（见第 8 节）。"
)
anchor5 = "%=====================================================================\n\\section{训练过程 Loss 与 Accuracy 曲线}"
assert anchor5 in src
src = src.replace(anchor5, env_new + "\n" + anchor5, 1)

# ---- 3) 第 8 章（不同模型的性能对比）末尾插入「多种子」----
seed_new = sec_seed.replace(
    "\\subsection{多种子重复实验}",
    "\\subsection{多种子重复实验与差异显著性}"
)
anchor8 = "%=====================================================================\n\\section{多模态对比实验}"
assert anchor8 in src
src = src.replace(anchor8, seed_new + "\n" + anchor8, 1)

# ---- 4) 第 9 章（多模态对比）末尾插入「推理剪枝」----
prune_new = r"""\subsection{模态依赖性：推理剪枝}

第 9.1 节的对比是"分别独立训练单模态模型"。另一个不同的问题是：\textbf{融合模型在实际推理时究竟用了哪一路？}
这可用推理剪枝直接检验——加载训练好的融合模型，推理时把某一路输入整段置零，观察预测如何变化。

\begin{table}[htbp]\centering
\caption{融合模型的推理剪枝结果（288 个测试样本）}
\label{tab:prune}
\begin{tabular}{lcccc}
\toprule
模型 & 完整融合 & 只留手套 & 只留 sEMG & 完整/只留手套 预测相同数 \\
\midrule
基线（种子 42） & 99.31\% & 99.31\% & 4.17\% & \textbf{288/288} \\
模态 Dropout 30\%（第 11 节） & 98.96\% & 98.96\% & 5.90\% & \textbf{288/288} \\
模态 Dropout 50\%（第 11 节） & 98.61\% & 98.61\% & 4.17\% & \textbf{288/288} \\
\bottomrule
\end{tabular}
\end{table}

结果出人意料：\textbf{把 sEMG 整段置零，模型预测逐样本一字不差（288/288）}；而把手套置零则退化为随机水平
（4.17\%，且 288 个样本只输出 1 个类别）。即\textbf{该"多模态融合"模型在功能上等价于纯手套单模态模型}，
sEMG 分支对最终决策的贡献精确为零。

这与第 9.1 节"单独训练 sEMG 可得 89.24\%"并不矛盾——信号本身携带信息，
但在融合架构下，当另一路已能近乎完美地解决问题时，优化器不会去利用它。第 11 节将系统检验这一判断。
"""
anchor9 = "%=====================================================================\n\\section{消融实验}"
assert anchor9 in src
src = src.replace(anchor9, prune_new + "\n" + anchor9, 1)

# ---- 5) 新增第 11 章「融合机制实验」（置于消融之后、测试集指标之前）----
fusion_intro = r"""%=====================================================================
\section{融合机制实验}
\label{sec:fusion}

第 8、9 节表明模型性能主要由手套模态决定，第 9.2 节的剪枝实验进一步显示 sEMG 分支完全未被使用。
本章从\textbf{训练机制}（模态 Dropout）与\textbf{信号处理}（滤波）两个方向系统检验"能否让融合真正生效"，
并用 Oracle 上界给出理论上的收益边界。

"""
sec_sum_new = sec_sum.replace("\\subsection{补充实验的综合结论}", "\\subsection{本章结论}")
fusion_block = fusion_intro + sec_drop + "\n" + sec_filt + "\n" + sec_orac + "\n" + sec_sum_new + "\n"
# 章内小节编号从 11.1 重排：把标题序号措辞理顺
fusion_block = fusion_block.replace("\\subsection{模态 Dropout：未能使 sEMG 被利用}",
                                    "\\subsection{模态 Dropout 实验}")
fusion_block = fusion_block.replace("\\subsection{sEMG 滤波对比：一个被推翻的直觉}",
                                    "\\subsection{sEMG 滤波对比实验}")
anchor11 = "%=====================================================================\n\\section{测试集评价指标}"
assert anchor11 in src
src = src.replace(anchor11, fusion_block + "\n" + anchor11, 1)

# ---- 6) 修掉正文里对旧「补充实验」的引用 ----
src = src.replace("见第~\\ref{sec:supp} 节", "见第 8 节")
src = src.replace("\\textbf{已补充 4 个随机种子的重复实验}：Bi-LSTM 为 $98.96\\pm0.40$\\,\\%，见第 8 节。",
                  "第 8 节对模型间差异采用 4 个随机种子的统计（Bi-LSTM $98.96\\pm0.40$\\,\\%）而非单次结果。")
src = src.replace("\\textbf{该推测已被补充实验推翻}：", "\\textbf{该推测已在第 11.2 节被实验推翻}：")
src = src.replace("改进方向不在输入侧，而在训练机制（见第~\\ref{sec:supp} 节末）。",
                  "改进方向不在输入侧，而在训练机制（见第 11 节）。")

open(P, "w", encoding="utf-8").write(src)
print("重构完成，新文件行数:", src.count("\n") + 1)
print()
for m in re.finditer(r"\\section\{([^}]*)\}", src):
    print("  S:", m.group(1))
