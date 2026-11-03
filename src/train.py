# -*- coding: utf-8 -*-
"""
阶段二：训练 / 评估主脚本（复现源论文双分支 Bi-LSTM）

示例:
    python src/train.py                                  # 默认：多模态融合
    python src/train.py --modality glove                 # 仅应变/压力（对比实验A）
    python src/train.py --modality semg                  # 仅 sEMG（对比实验B）
    python src/train.py --lr 0.0005 --batch-size 32 --tag lr5e-4   # 调参实验
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import build_dataloaders              # noqa: E402
from src.models.bilstm import BiLSTMFusion                  # noqa: E402
from src.models.others import build_model_by_name           # noqa: E402
from src.utils import (set_seed, get_device, evaluate, compute_metrics,
                       plot_curves, plot_confusion)          # noqa: E402


def load_config(path):
    """加载 YAML 配置。

    paths 下的相对路径基于仓库根目录解析（即 config.yaml 所在目录的上一级）；
    也可用环境变量覆盖为绝对路径：SIGNLANG_DATA_ROOT / SIGNLANG_CACHE_DIR /
    SIGNLANG_CKPT_DIR / SIGNLANG_FIG_DIR / SIGNLANG_RES_DIR。
    """
    import yaml
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    env_keys = {"data_root": "SIGNLANG_DATA_ROOT", "cache_dir": "SIGNLANG_CACHE_DIR",
                "ckpt_dir": "SIGNLANG_CKPT_DIR", "fig_dir": "SIGNLANG_FIG_DIR",
                "res_dir": "SIGNLANG_RES_DIR"}
    for key, env in env_keys.items():
        val = os.environ.get(env) or cfg["paths"].get(key)
        cfg["paths"][key] = val if os.path.isabs(val) else os.path.join(root, val)
    return cfg


def train_one(cfg, overrides=None):
    ov = overrides or {}
    seed = ov.get("seed", cfg.get("seed", 42))
    set_seed(seed)
    device = get_device(cfg.get("device", "cuda"))
    cache = ov.get("cache_dir", cfg["paths"]["cache_dir"])
    mod_drop = ov.get("modality_dropout", 0.0)   # 训练时随机置零一路，强制两支都学
    tag = ov.get("tag", "bilstm_fusion")
    modality = ov.get("modality", "both")
    model_name = ov.get("model", "bilstm")

    bs = ov.get("batch_size", cfg["train"]["batch_size"])
    epochs = ov.get("epochs", cfg["train"]["epochs"])
    lr = ov.get("lr", cfg["train"]["lr"])
    wd = ov.get("weight_decay", cfg["train"]["weight_decay"])
    dropout = ov.get("dropout", cfg["train"]["dropout"])
    patience = ov.get("patience", cfg["train"]["early_stop_patience"])
    aug = ov.get("augment", False)

    aug_cfg = {"noise_std": 0.02, "scale_prob": 0.5, "scale_range": 0.2,
               "chan_drop_prob": 0.3, "chan_drop_frac": 0.1,
               "shift_prob": 0.3, "shift_max": 20}
    dl_tr, dl_va, dl_te, info = build_dataloaders(
        cache, batch_size=bs, augment_train=aug, aug_cfg=aug_cfg, seed=seed)
    n_classes = info["num_classes"]
    names = [None] * n_classes
    for k, v in info["label_map"].items():
        names[v] = k

    model = build_model_by_name(model_name, cfg["model"], dropout=dropout,
                                modality=modality, **ov.get("model_kw", {})).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] model={model_name} modality={modality} 参数量={n_param:,} 设备={device} "
          f"train/val/test={info['n_train']}/{info['n_val']}/{info['n_test']}")

    crit = nn.CrossEntropyLoss(label_smoothing=ov.get("label_smoothing", 0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # 混合精度 + 梯度累积：论文 batch=50 在 8GB 卡上直接用 fp32 会 OOM(10.7GB)，
    # 开 AMP(fp16) 后 batch=50 峰值仅 ~5.5GB，可精确复现论文 batch。
    use_amp = ov.get("amp", cfg["train"].get("amp", True)) and device.type == "cuda"
    accum = ov.get("accum", cfg["train"].get("accum", 1))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_dir = cfg["paths"]["ckpt_dir"]; os.makedirs(ckpt_dir, exist_ok=True)
    res_dir = cfg["paths"]["res_dir"]; os.makedirs(res_dir, exist_ok=True)
    fig_dir = cfg["paths"]["fig_dir"]; os.makedirs(fig_dir, exist_ok=True)
    log_dir = os.path.join(os.path.dirname(res_dir), "logs"); os.makedirs(log_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, f"{tag}.pt")
    # 逐 epoch 进度落盘（stdout 被管道缓冲时也能实时监控）
    prog_path = os.path.join(log_dir, f"{tag}_progress.log")
    with open(prog_path, "w", encoding="utf-8") as fp:
        fp.write(f"# tag={tag} model={model_name} modality={modality} started={time.strftime('%F %T')}\n")

    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "lr"]}
    best_val, best_ep, wait = -1.0, -1, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tl, tn, tc = 0.0, 0, 0
        n_batch = len(dl_tr)
        for i, (glove, semg, y) in enumerate(dl_tr):
            glove, semg, y = glove.to(device), semg.to(device), y.to(device)
            if mod_drop > 0 and modality == "both":
                # 模态 dropout：每条样本以 mod_drop 概率随机丢一路（各占一半）
                r = torch.rand(glove.size(0), device=device)
                drop_g = r < mod_drop * 0.5
                drop_s = (r >= mod_drop * 0.5) & (r < mod_drop)
                if drop_g.any():
                    glove = glove.clone(); glove[drop_g] = 0.0
                if drop_s.any():
                    semg = semg.clone(); semg[drop_s] = 0.0
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                out = model(glove, semg)
                loss = crit(out, y) / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0 or (i + 1) == n_batch:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            tl += loss.item() * accum * len(y); tn += len(y)
            tc += (out.argmax(1) == y).sum().item()
        sched.step()
        yv, pv, _, vl = evaluate(model, dl_va, device)
        va = (yv == pv).mean()
        hist["train_loss"].append(tl / tn); hist["train_acc"].append(tc / tn)
        hist["val_loss"].append(vl); hist["val_acc"].append(float(va))
        hist["lr"].append(opt.param_groups[0]["lr"])
        flag = ""
        if va > best_val:
            best_val, best_ep, wait = float(va), ep, 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "modality": modality,
                        "model_name": model_name,
                        "epoch": ep, "val_acc": float(va), "label_map": info["label_map"]},
                       best_path)
            flag = " *best"
        else:
            wait += 1
        print(f"  ep{ep:3d} | train_loss {tl/tn:.4f} acc {tc/tn:.4f} "
              f"| val_loss {vl:.4f} acc {va:.4f}{flag}", flush=True)
        with open(prog_path, "a", encoding="utf-8") as fp:
            fp.write(f"ep{ep:3d} train_loss={tl/tn:.4f} train_acc={tc/tn:.4f} "
                     f"val_loss={vl:.4f} val_acc={va:.4f}{flag}\n")
        if wait >= patience:
            print(f"  早停于 epoch {ep}（{patience} 轮无提升）")
            break
    train_time = time.time() - t0

    # ---- 载入最优权重，测试集评估 ----
    ck = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    yt, yp, yprob, _ = evaluate(model, dl_te, device)
    met = compute_metrics(yt, yp, n_classes)
    print(f"\n[{tag}] 测试集 Accuracy={met['accuracy']:.4f} "
          f"Precision(macro)={met['precision_macro']:.4f} "
          f"F1(macro)={met['f1_macro']:.4f} | 最优epoch={best_ep} 用时={train_time:.1f}s")

    plot_curves(hist, os.path.join(fig_dir, f"{tag}_curves.png"), title=f"{tag} ({modality})")
    plot_confusion(met["confusion_matrix"], names, os.path.join(fig_dir, f"{tag}_cm.png"),
                   title=f"Confusion Matrix - {tag}")
    pd.DataFrame(hist).to_csv(os.path.join(res_dir, f"{tag}_history.csv"), index=False)
    pd.DataFrame({"y_true": yt, "y_pred": yp,
                  "y_true_name": [names[i] for i in yt],
                  "y_pred_name": [names[i] for i in yp]}).to_csv(
        os.path.join(res_dir, f"{tag}_test_pred.csv"), index=False, encoding="utf-8-sig")

    summary = {"tag": tag, "model": model_name, "modality": modality, "n_params": n_param,
               "best_epoch": best_ep, "best_val_acc": best_val,
               "test_accuracy": met["accuracy"],
               "test_precision_macro": met["precision_macro"],
               "test_recall_macro": met["recall_macro"],
               "test_f1_macro": met["f1_macro"],
               "test_precision_weighted": met["precision_weighted"],
               "train_time_sec": train_time,
               "config": {"batch_size": bs, "epochs": epochs, "lr": lr,
                          "weight_decay": wd, "dropout": dropout, "augment": aug}}
    with open(os.path.join(res_dir, f"{tag}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config.yaml"))
    ap.add_argument("--modality", default="both", choices=["both", "glove", "semg"])
    ap.add_argument("--model", default="bilstm",
                    choices=["bilstm", "cnn", "gru", "tcn", "transformer"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cache-dir", default=None, help="覆盖缓存目录（用于预处理变体对比）")
    ap.add_argument("--modality-dropout", type=float, default=0.0,
                    help="训练时随机置零一路输入的概率（0~1），用于强制两支都被利用")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    ov = {"modality": args.modality, "augment": args.augment, "model": args.model}
    ov["tag"] = args.tag or f"{args.model}_{args.modality}"
    if args.cache_dir:
        ov["cache_dir"] = args.cache_dir
    if args.modality_dropout > 0:
        ov["modality_dropout"] = args.modality_dropout
    for k, v in [("epochs", args.epochs), ("batch_size", args.batch_size),
                 ("lr", args.lr), ("dropout", args.dropout), ("weight_decay", args.weight_decay)]:
        if v is not None:
            ov[k] = v
    train_one(cfg, ov)


if __name__ == "__main__":
    main()
