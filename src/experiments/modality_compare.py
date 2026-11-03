# -*- coding: utf-8 -*-
"""
实验要求 4：多模态对比实验
A 只用应变/压力(手套)  B 只用 sEMG  C 两者融合

用法: python src/experiments/modality_compare.py --epochs 100
产物: outputs/results/modality_compare.csv   outputs/figures/modality_compare.png
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.experiments.common import (default_cfg, run_grid, save_results,   # noqa: E402
                                    plot_metric_bar)

NAME_CN = {"glove": "A 仅应变/压力(手套)", "semg": "B 仅 sEMG", "both": "C 应变/压力+sEMG 融合"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--model", default="bilstm")
    args = ap.parse_args()

    cfg = default_cfg()
    os.makedirs(cfg["paths"]["res_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["fig_dir"], exist_ok=True)

    ovs = []
    for mod in ["glove", "semg", "both"]:
        ov = {"modality": mod, "model": args.model, "tag": f"modality_{mod}"}
        if args.epochs:
            ov["epochs"] = args.epochs
        ovs.append(ov)

    res = run_grid(cfg, ovs, label="modality-compare")
    df = save_results(res, os.path.join(cfg["paths"]["res_dir"], "modality_compare.csv"))
    df["实验"] = [NAME_CN.get(m, m) for m in df["modality"]]
    plot_metric_bar(df, "实验", "test_accuracy",
                    os.path.join(cfg["paths"]["fig_dir"], "modality_compare.png"),
                    title="多模态对比：单模态 vs 多模态融合（测试集准确率）")

    print("\n=== 多模态对比结论 ===")
    for _, r in df.iterrows():
        print(f"  {r['实验']:24s} test_acc={r['test_accuracy']:.4f}  "
              f"precision={r['test_precision_macro']:.4f}")


if __name__ == "__main__":
    main()
