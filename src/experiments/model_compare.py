# -*- coding: utf-8 -*-
"""
实验要求 3：不同模型对比实验（主模型 + ≥3 种其他模型）

用法:
    python src/experiments/model_compare.py                        # 全部 5 个模型
    python src/experiments/model_compare.py --models bilstm,cnn,gru,tcn
产物: outputs/results/model_compare.csv   outputs/figures/model_compare.png
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.experiments.common import (default_cfg, run_grid, save_results,   # noqa: E402
                                    plot_metric_bar)

NAME_CN = {"bilstm": "Bi-LSTM（主模型）", "cnn": "1D-CNN", "gru": "Bi-GRU",
           "tcn": "TCN", "transformer": "Transformer"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="bilstm,cnn,gru,tcn")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--modality", default="both")
    args = ap.parse_args()

    cfg = default_cfg()
    os.makedirs(cfg["paths"]["res_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["fig_dir"], exist_ok=True)

    ovs = []
    for m in args.models.split(","):
        ov = {"model": m, "modality": args.modality, "tag": f"model_{m}"}
        if args.epochs:
            ov["epochs"] = args.epochs
        ovs.append(ov)

    res = run_grid(cfg, ovs, label="model-compare")
    df = save_results(res, os.path.join(cfg["paths"]["res_dir"], "model_compare.csv"))
    df["模型"] = [NAME_CN.get(m, m) for m in df["model"]]
    df = df.sort_values("test_accuracy", ascending=False)
    plot_metric_bar(df, "模型", "test_accuracy",
                    os.path.join(cfg["paths"]["fig_dir"], "model_compare.png"),
                    title="不同模型性能对比（测试集准确率）")

    print("\n=== 模型对比排名 ===")
    for i, (_, r) in enumerate(df.iterrows(), 1):
        print(f"  {i}. {r['模型']:18s} acc={r['test_accuracy']:.4f}  "
              f"F1={r['test_f1_macro']:.4f}  params={r['n_params']:,}")


if __name__ == "__main__":
    main()
