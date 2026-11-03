# -*- coding: utf-8 -*-
"""对指定检查点做测试时模态剪枝分析。

用法: python tools/pruning_check.py <tag> [<tag> ...]
输出每个检查点的：完整 / 只留手套 / 只留 sEMG 的测试准确率，以及预测一致性。
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
from src.utils import set_seed                           # noqa: E402
from src.train import load_config                        # noqa: E402

CFG = os.path.join(ROOT, "configs", "config_remote.yaml")
cfg = load_config(CFG)
set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

tags = sys.argv[1:]
if not tags:
    sys.exit("用法: python tools/pruning_check.py <tag> [...]")

out = {}
for tag in tags:
    ck_path = os.path.join(cfg["paths"]["ckpt_dir"], f"{tag}.pt")
    if not os.path.exists(ck_path):
        print(f"[{tag}] 检查点不存在，跳过")
        continue
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    modality = ck.get("modality", "both")
    model_name = ck.get("model_name", "bilstm")
    # 剪枝实验只在融合模型上有意义
    if modality != "both":
        print(f"[{tag}] modality={modality}，非融合模型，跳过剪枝")
        continue

    # 关键：被剪枝的模型必须在与它训练时一致的数据分布上评估。
    # 滤波版模型（p2_filter_*）训练自 cache/filtered，用基础缓存评它会分布错配。
    cache_dir = cfg["paths"]["cache_dir"]
    for a in sys.argv[1:]:
        if a.startswith("--cache-dir="):
            cache_dir = a.split("=", 1)[1]
        elif a == "--cache-dir" and sys.argv.index(a) + 1 < len(sys.argv):
            cache_dir = sys.argv[sys.argv.index(a) + 1]
    print(f"[{tag}] cache_dir = {cache_dir}")
    _, _, dl_te, info = build_dataloaders(cache_dir, batch_size=64,
                                          num_workers=0, augment_train=False)
    model = BiLSTMFusion(cfg["model"]["glove"], cfg["model"]["semg"],
                         num_classes=len(info["label_map"]),
                         dropout=cfg["train"]["dropout"], modality="both").to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    def run(zg=False, zs=False):
        P = []
        with torch.no_grad():
            for g, s, y in dl_te:
                g, s = g.to(device), s.to(device)
                if zg:
                    g = torch.zeros_like(g)
                if zs:
                    s = torch.zeros_like(s)
                P.append(model(g, s).argmax(1).cpu().numpy())
        return np.concatenate(P)

    y = []
    with torch.no_grad():
        for g, s, yy in dl_te:
            y.append(yy.numpy())
    y = np.concatenate(y)

    p_full, p_g, p_s = run(), run(zs=True), run(zg=True)
    a_full = float((p_full == y).mean())
    a_g = float((p_g == y).mean())
    a_s = float((p_s == y).mean())
    same = int((p_full == p_g).sum())
    n_uniq = int(len(np.unique(p_s)))

    print(f"[{tag}]  (ckpt epoch={ck.get('epoch')})")
    print(f"    完整融合        = {a_full*100:6.2f}%")
    print(f"    只留手套(sEMG=0)= {a_g*100:6.2f}%   Δ={(a_g-a_full)*100:+.2f}pp"
          f"   预测完全相同 {same}/288")
    print(f"    只留sEMG(手套=0)= {a_s*100:6.2f}%   Δ={(a_s-a_full)*100:+.2f}pp"
          f"   输出唯一类别数 {n_uniq}")
    out[tag] = {"full": a_full, "glove_only": a_g, "semg_only": a_s,
                "identical_pred_full_vs_gloveonly": same, "unique_classes_semg_only": n_uniq}

with open(os.path.join(cfg["paths"]["res_dir"], "pruning_check.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\n已保存 outputs/results/pruning_check.json")
