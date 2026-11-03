# -*- coding: utf-8 -*-
"""
实验要求 2：参数调节实验
对 2~4 个关键参数逐个扫描，记录验证集/测试集准确率，绘制折线图与汇总表，给出最优组合。

用法:
    python src/experiments/param_sweep.py                              # 默认扫 lr/batch_size/dropout
    python src/experiments/param_sweep.py --params lr,dropout --epochs 60
    python src/experiments/param_sweep.py --param lr --values 0.0005,0.001,0.002

产物: outputs/results/param_sweep_<param>.csv  outputs/figures/param_sweep_<param>.png
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.experiments.common import (default_cfg, run_grid, save_results,   # noqa: E402
                                    plot_param_line)

GRIDS = {
    "lr": [0.0005, 0.001, 0.002, 0.005],
    "batch_size": [16, 32, 50],
    "dropout": [0.1, 0.3, 0.5],
    "weight_decay": [0.0, 0.005, 0.05],
}
BASE = {"lr": 0.001, "batch_size": 50, "dropout": 0.3, "weight_decay": 0.005}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="lr,batch_size,dropout")
    ap.add_argument("--param", default=None, help="只扫单个参数")
    ap.add_argument("--values", default=None, help="逗号分隔的自定义取值")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--modality", default="both")
    ap.add_argument("--model", default="bilstm")
    args = ap.parse_args()

    cfg = default_cfg()
    params = [args.param] if args.param else args.params.split(",")
    os.makedirs(cfg["paths"]["res_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["fig_dir"], exist_ok=True)

    all_rows = []
    for p in params:
        vals = ([float(v) if "." in v else int(v) for v in args.values.split(",")]
                if args.values else GRIDS[p])
        ovs = []
        for v in vals:
            ov = {"modality": args.modality, "model": args.model, **BASE}
            ov[p] = v
            ov["tag"] = f"sweep_{p}={v}"
            if args.epochs:
                ov["epochs"] = args.epochs
            ovs.append(ov)
        res = run_grid(cfg, ovs, label=f"sweep-{p}")
        df = save_results(res, os.path.join(cfg["paths"]["res_dir"], f"param_sweep_{p}.csv"))
        df["param"] = p
        all_rows.append(df)
        plot_param_line(df.rename(columns={p: "value"}) if p in df.columns else df,
                        p if p in df.columns else "cfg_" + p,
                        os.path.join(cfg["paths"]["fig_dir"], f"param_sweep_{p}.png"))
        # 打印该参数最优
        best = df.loc[df["best_val_acc"].idxmax()]
        print(f"\n>>> 参数 {p} 最优取值 = {best.get(p, best.get('cfg_' + p))} "
              f"(val={best['best_val_acc']:.4f}, test={best['test_accuracy']:.4f})")

    import pandas as pd
    merged = pd.concat(all_rows, ignore_index=True)
    merged.to_csv(os.path.join(cfg["paths"]["res_dir"], "param_sweep_all.csv"),
                  index=False, encoding="utf-8-sig")
    print("\n汇总已保存: outputs/results/param_sweep_all.csv")


if __name__ == "__main__":
    main()
