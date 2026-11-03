# -*- coding: utf-8 -*-
"""
阶段一：PyTorch Dataset / DataLoader
读取 preprocess.py 产出的缓存，供训练阶段使用。

用法:
    from src.data.dataset import SignLanguageDataset, build_dataloaders
    train_dl, val_dl, test_dl, info = build_dataloaders(cache_dir, batch_size=50)
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SignLanguageDataset(Dataset):
    """返回 (glove, semg, label)。
    glove: (400, 40) float32 ; semg: (1950, 20) float32 ; label: int64
    augment=True 时施加轻量数据增强（仅用于训练集）。
    """

    def __init__(self, X_glove, X_semg, y, augment=False, aug_cfg=None):
        self.Xg = torch.from_numpy(np.ascontiguousarray(X_glove)).float()
        self.Xs = torch.from_numpy(np.ascontiguousarray(X_semg)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(y)).long()
        self.augment = augment
        self.cfg = aug_cfg or {}
        assert len(self.Xg) == len(self.Xs) == len(self.y)

    def __len__(self):
        return len(self.y)

    def _augment(self, g, s):
        cfg = self.cfg
        # 1) 高斯噪声
        ns = cfg.get("noise_std", 0.0)
        if ns > 0:
            g = g + torch.randn_like(g) * ns
            s = s + torch.randn_like(s) * ns
        # 2) 幅值缩放
        if cfg.get("scale_prob", 0.0) > 0 and torch.rand(1).item() < cfg["scale_prob"]:
            a = 1.0 + (torch.rand(1).item() - 0.5) * cfg.get("scale_range", 0.2)
            g = g * a
            s = s * a
        # 3) 通道丢包（模拟传感器瞬时失效）
        if cfg.get("chan_drop_prob", 0.0) > 0 and torch.rand(1).item() < cfg["chan_drop_prob"]:
            for x, c in ((g, g.shape[1]), (s, s.shape[1])):
                m = (torch.rand(c) > cfg.get("chan_drop_frac", 0.1)).float()
                x.mul_(m)
        # 4) 时间平移（循环滚动，长度不变）
        if cfg.get("shift_prob", 0.0) > 0 and torch.rand(1).item() < cfg["shift_prob"]:
            k = int(torch.randint(-cfg.get("shift_max", 20), cfg.get("shift_max", 20) + 1, (1,)).item())
            if k:
                g = torch.roll(g, shifts=k, dims=0)
                s = torch.roll(s, shifts=k, dims=0)
        return g, s

    def __getitem__(self, i):
        g, s, y = self.Xg[i], self.Xs[i], self.y[i]
        if self.augment:
            g, s = self._augment(g, s)
        return g, s, y


def load_cache(cache_dir):
    Xg = np.load(os.path.join(cache_dir, "X_glove.npy"))
    Xs = np.load(os.path.join(cache_dir, "X_semg.npy"))
    y = np.load(os.path.join(cache_dir, "y.npy"))
    with open(os.path.join(cache_dir, "label_map.json"), encoding="utf-8") as f:
        label_map = json.load(f)
    import pandas as pd
    meta = pd.read_csv(os.path.join(cache_dir, "meta.csv"))
    return Xg, Xs, y, label_map, meta


def build_dataloaders(cache_dir, batch_size=50, num_workers=0,
                      augment_train=True, aug_cfg=None, seed=42):
    Xg, Xs, y, label_map, meta = load_cache(cache_dir)
    split = meta["split"].values
    tr, va, te = split == "train", split == "val", split == "test"

    ds_tr = SignLanguageDataset(Xg[tr], Xs[tr], y[tr], augment=augment_train, aug_cfg=aug_cfg)
    ds_va = SignLanguageDataset(Xg[va], Xs[va], y[va], augment=False)
    ds_te = SignLanguageDataset(Xg[te], Xs[te], y[te], augment=False)

    g = torch.Generator().manual_seed(seed)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, generator=g, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    info = {
        "num_classes": len(label_map), "label_map": label_map,
        "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
        "glove_shape": tuple(Xg.shape[1:]), "semg_shape": tuple(Xs.shape[1:]),
    }
    return dl_tr, dl_va, dl_te, info


if __name__ == "__main__":
    CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "outputs", "cache")
    dl_tr, dl_va, dl_te, info = build_dataloaders(CACHE, batch_size=50)
    print("info:", {k: v for k, v in info.items() if k != "label_map"})
    print("类别数:", info["num_classes"])
    g, s, y = next(iter(dl_tr))
    print("一个 batch:", g.shape, s.shape, y.shape, "| dtype:", g.dtype, s.dtype, y.dtype)
    print("label 取值范围:", int(y.min()), "-", int(y.max()))
