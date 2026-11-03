# -*- coding: utf-8 -*-
"""
实验要求 5：消融实验
把主模型（双分支 Bi-LSTM 融合）中的某个模块去掉/替换，重训看性能是否下降。

消融项:
  A0 完整模型（baseline）
  A1 去掉分支内的 2 层 FC（用 Bi-LSTM 末时刻特征直接融合）
  A2 双向 → 单向（去掉 Bi-LSTM 的反向链路）
  A3 堆叠层数 3 → 1（减少时序深度）
  A4 融合方式 concat → sum（改变特征融合算子）

用法: python src/experiments/ablation.py --epochs 100
产物: outputs/results/ablation.csv   outputs/figures/ablation.png
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.experiments.common import (default_cfg, run_grid, save_results,   # noqa: E402
                                    plot_metric_bar)


def variants():
    return [
        {"tag": "abl_A0_full", "model_kw": {}},
        {"tag": "abl_A1_no_fc", "model_kw": {"use_fc": False}},
        {"tag": "abl_A2_uni_lstm", "model_kw": {"bidirectional": False}},
        {"tag": "abl_A3_layers1", "model_kw": {},
         "cfg_patch": {"model.glove.num_layers": 1, "model.semg.num_layers": 1}},
        {"tag": "abl_A4_fusion_sum", "model_kw": {"fusion": "sum"}},
    ]


NAME_CN = {
    "abl_A0_full": "A0 完整模型(基准)",
    "abl_A1_no_fc": "A1 去掉分支内 FC",
    "abl_A2_uni_lstm": "A2 双向→单向 LSTM",
    "abl_A3_layers1": "A3 层数 3→1",
    "abl_A4_fusion_sum": "A4 融合 concat→sum",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--modality", default="both")
    args = ap.parse_args()

    cfg = default_cfg()
    os.makedirs(cfg["paths"]["res_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["fig_dir"], exist_ok=True)

    ovs = []
    for v in variants():
        ov = {"model": "bilstm", "modality": args.modality, **v}
        if args.epochs:
            ov["epochs"] = args.epochs
        ovs.append(ov)

    res = run_grid(cfg, ovs, label="ablation")
    df = save_results(res, os.path.join(cfg["paths"]["res_dir"], "ablation.csv"))
    df["消融项"] = [NAME_CN.get(t, t) for t in df["tag"]]
    plot_metric_bar(df, "消融项", "test_accuracy",
                    os.path.join(cfg["paths"]["fig_dir"], "ablation.png"),
                    title="消融实验（测试集准确率）")

    base = df.loc[df["tag"] == "abl_A0_full", "test_accuracy"]
    print("\n=== 消融结果（相对基准的下降）===")
    b = float(base.iloc[0]) if len(base) else float("nan")
    for _, r in df.iterrows():
        d = r["test_accuracy"] - b
        print(f"  {r['消融项']:22s} acc={r['test_accuracy']:.4f}  Δ={d:+.4f}")


if __name__ == "__main__":
    main()
