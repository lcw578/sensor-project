# -*- coding: utf-8 -*-
"""测试时模态剪枝实验（无需训练）

验证假说：源论文的单模态数字（仅手套 83.75% / 仅 sEMG 88.02%）是否来自
「在训练好的融合模型上做测试时模态剪枝」，而非「单独训练单模态模型」。

做法：加载已训练好的融合模型权重，推理时把另一路输入置零，测同一测试集准确率。
对照：
  - 本复现的「独立训练单模态模型」：仅手套 98.26% / 仅 sEMG 91.32%（前已测得）
  - 源论文：仅手套 83.75% / 仅 sEMG 88.02%
"""
import os
import sys
import json
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data.dataset import build_dataloaders          # noqa: E402
from src.models.bilstm import BiLSTMFusion               # noqa: E402
from src.utils import set_seed                        # noqa: E402
from src.train import load_config                      # noqa: E402

cfg = load_config(os.path.join(ROOT, "configs", "config.yaml"))
set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

_, _, dl_te, info = build_dataloaders(cfg["paths"]["cache_dir"], batch_size=64,
                                      num_workers=0, augment_train=False)
n_cls = len(info["label_map"])

model = BiLSTMFusion(cfg["model"]["glove"], cfg["model"]["semg"],
                     num_classes=n_cls, dropout=cfg["train"]["dropout"],
                     modality="both").to(device)
ck = torch.load(os.path.join(cfg["paths"]["ckpt_dir"], "bilstm_both.pt"),
                map_location=device, weights_only=False)
model.load_state_dict(ck["model"])
model.eval()
print(f"加载检查点: epoch={ck['epoch']} val_acc={float(ck['val_acc']):.4f}\n")


def evaluate_with_mask(zero_glove=False, zero_semg=False):
    ys, ps = [], []
    with torch.no_grad():
        for g, s, y in dl_te:
            g, s = g.to(device), s.to(device)
            if zero_glove:
                g = torch.zeros_like(g)
            if zero_semg:
                s = torch.zeros_like(s)
            logits = model(g, s)
            ps.append(logits.argmax(1).cpu().numpy())
            ys.append(y.numpy())
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    return ys, ps


def acc(ys, ps):
    return float((ys == ps).mean())


rows = []
ys, ps = evaluate_with_mask()
a_both = acc(ys, ps)
rows.append(("完整融合输入（对照）", a_both, None))

ys, ps = evaluate_with_mask(zero_semg=True)
a_g = acc(ys, ps)
rows.append(("测试时剪枝：只留手套（sEMG 置零）", a_g, a_g - a_both))

ys, ps = evaluate_with_mask(zero_glove=True)
a_s = acc(ys, ps)
rows.append(("测试时剪枝：只留 sEMG（手套置零）", a_s, a_s - a_both))

print("=" * 78)
print("测试时模态剪枝结果（同一测试集 288 样本）")
print("=" * 78)
for name, a, d in rows:
    ds = f"  Δ={d*100:+.2f}pp" if d is not None else ""
    print(f"  {name:38s} Accuracy = {a*100:6.2f}%{ds}")

print()
print("=" * 78)
print("与「独立训练单模态模型」及源论文的对照")
print("=" * 78)
indep = json.load(open(os.path.join(ROOT, "outputs/results/modality_glove_summary.json"),
                       encoding="utf-8"))
indep_s = json.load(open(os.path.join(ROOT, "outputs/results/modality_semg_summary.json"),
                         encoding="utf-8"))
print(f"{'':30s}{'仅手套':>12s}{'仅 sEMG':>12s}")
print(f"{'源论文':30s}{'83.75%':>12s}{'88.02%':>12s}")
print(f"{'本复现 · 独立训练单模态':30s}{indep['test_accuracy']*100:11.2f}%{indep_s['test_accuracy']*100:11.2f}%")
print(f"{'本复现 · 测试时模态剪枝':30s}{a_g*100:11.2f}%{a_s*100:11.2f}%")
print()
dg = abs(a_g * 100 - 83.75)
ds_ = abs(a_s * 100 - 88.02)
print(f"剪枝版与论文的差距: 仅手套 {dg:.2f}pp   仅 sEMG {ds_:.2f}pp")
print(f"独立训练版与论文的差距: 仅手套 {abs(indep['test_accuracy']*100-83.75):.2f}pp   "
      f"仅 sEMG {abs(indep_s['test_accuracy']*100-88.02):.2f}pp")

out = {"both": a_both, "prune_to_glove": a_g, "prune_to_semg": a_s,
       "indep_glove": indep["test_accuracy"], "indep_semg": indep_s["test_accuracy"],
       "prune_glove_gap_to_paper_pp": dg, "prune_semg_gap_to_paper_pp": ds_}
with open(os.path.join(ROOT, "outputs/results/inference_pruning.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\n已保存 outputs/results/inference_pruning.json")
