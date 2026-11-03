# -*- coding: utf-8 -*-
"""从已有结果 CSV 重绘汇总柱状图（不重训），用于修正排版后重出图。"""
import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.experiments.common import plot_metric_bar   # noqa: E402

RES = os.path.join(ROOT, "outputs", "results")
FIG = os.path.join(ROOT, "outputs", "figures")

MOD_CN = {"glove": "A 仅应变/压力", "semg": "B 仅 sEMG", "both": "C 融合（应变/压力+sEMG）"}
NAME_CN = {"bilstm": "Bi-LSTM（主模型）", "cnn": "1D-CNN", "gru": "Bi-GRU", "tcn": "TCN"}
ABL_CN = {"abl_A0_full": "A0 完整模型(基准)", "abl_A1_no_fc": "A1 去掉分支内 FC",
          "abl_A2_uni_lstm": "A2 双向→单向", "abl_A3_layers1": "A3 层数 3→1",
          "abl_A4_fusion_sum": "A4 融合 concat→sum"}

# 多模态
d = pd.read_csv(f"{RES}/modality_compare.csv")
d["实验"] = [MOD_CN.get(m, m) for m in d["modality"]]
plot_metric_bar(d, "实验", "test_accuracy", f"{FIG}/modality_compare.png",
                title="多模态对比：单模态 vs 多模态融合（测试集准确率）")

# 模型
d = pd.read_csv(f"{RES}/model_compare.csv").sort_values("test_accuracy", ascending=False)
d["模型"] = [NAME_CN.get(m, m) for m in d["model"]]
plot_metric_bar(d, "模型", "test_accuracy", f"{FIG}/model_compare.png",
                title="不同模型性能对比（测试集准确率）")

# 消融
d = pd.read_csv(f"{RES}/ablation.csv")
d["消融项"] = [ABL_CN.get(t, t) for t in d["tag"]]
plot_metric_bar(d, "消融项", "test_accuracy", f"{FIG}/ablation.png",
                title="消融实验（测试集准确率）")

print("重绘完成")
