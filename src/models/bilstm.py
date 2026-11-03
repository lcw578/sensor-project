# -*- coding: utf-8 -*-
"""
主模型：双分支 Bi-LSTM 特征融合（对齐源论文 Figure 7c）

    手套(应变/压力) (B,400,40) ─→ 3层 Bi-LSTM(hidden=128) ─→ 2层 FC ─┐
                                                                    ├─→ concat ─→ FC ─→ 24类
    sEMG           (B,1950,20) ─→ 3层 Bi-LSTM(hidden=512) ─→ 2层 FC ─┘

- 每个 Branch 内：Bi-LSTM 取最后时刻隐状态（前向+后向拼接, 2*hidden）→ 2 层 FC。
- 支持 modality ∈ {both, glove, semg}，用于多模态对比实验 A/B/C。
- 支持 branch 内去掉 FC（消融）: use_fc=False。
- 支持 pooling ∈ {last, mean}。
"""
import torch
import torch.nn as nn


class BiLSTMBranch(nn.Module):
    """单模态分支：N 层 Bi-LSTM + 2 层 FC。"""

    def __init__(self, input_dim, hidden, num_layers=3, fc=(256, 128),
                 dropout=0.3, use_fc=True, pooling="last", bidirectional=True):
        super().__init__()
        self.pooling = pooling
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = hidden * (2 if bidirectional else 1)
        self.use_fc = use_fc
        if use_fc:
            layers = []
            dims = [self.out_dim] + list(fc)
            for i in range(len(fc)):
                layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(inplace=True),
                           nn.Dropout(dropout)]
            self.fc = nn.Sequential(*layers)
            self.out_dim = fc[-1]
        else:
            self.fc = None

    def forward(self, x):
        # x: (B, T, C)
        out, _ = self.lstm(x)                       # (B, T, 2H)
        if self.pooling == "mean":
            feat = out.mean(dim=1)
        else:
            feat = out[:, -1, :]                    # 最后时刻
        if self.fc is not None:
            feat = self.fc(feat)
        return feat


class BiLSTMFusion(nn.Module):
    """双分支 Bi-LSTM 特征融合分类器。"""

    def __init__(self, glove_cfg, semg_cfg, num_classes=24, dropout=0.3,
                 modality="both", use_fc=True, pooling="last",
                 fusion="concat", bidirectional=True):
        super().__init__()
        assert modality in ("both", "glove", "semg")
        assert fusion in ("concat", "sum")
        self.modality = modality
        self.fusion = fusion
        self.glove_branch = BiLSTMBranch(
            glove_cfg["input_dim"], glove_cfg["hidden"], glove_cfg.get("num_layers", 3),
            tuple(glove_cfg.get("fc", (256, 128))), dropout, use_fc, pooling, bidirectional)
        self.semg_branch = BiLSTMBranch(
            semg_cfg["input_dim"], semg_cfg["hidden"], semg_cfg.get("num_layers", 3),
            tuple(semg_cfg.get("fc", (256, 128))), dropout, use_fc, pooling, bidirectional)

        if modality == "both":
            if fusion == "sum":
                assert self.glove_branch.out_dim == self.semg_branch.out_dim, \
                    "sum 融合要求两个分支输出维度相同"
                fused = self.glove_branch.out_dim
            else:
                fused = self.glove_branch.out_dim + self.semg_branch.out_dim
        elif modality == "glove":
            fused = self.glove_branch.out_dim
        else:
            fused = self.semg_branch.out_dim
        self.fused_dim = fused
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused, num_classes),
        )

    def forward(self, glove, semg):
        feats = []
        if self.modality in ("both", "glove"):
            feats.append(self.glove_branch(glove))
        if self.modality in ("both", "semg"):
            feats.append(self.semg_branch(semg))
        if len(feats) == 1:
            z = feats[0]
        elif self.fusion == "sum":
            z = feats[0] + feats[1]
        else:
            z = torch.cat(feats, dim=1)
        return self.head(z)


def build_model(cfg, modality="both", **kw):
    """从 config dict 构建模型。cfg 为 configs/config.yaml 的 model 段。"""
    return BiLSTMFusion(
        glove_cfg=cfg["glove"], semg_cfg=cfg["semg"],
        num_classes=cfg.get("num_classes", 24),
        dropout=kw.pop("dropout", 0.3),
        modality=modality, **kw,
    )


if __name__ == "__main__":
    m = BiLSTMFusion({"input_dim": 40, "hidden": 128, "num_layers": 3, "fc": [256, 128]},
                     {"input_dim": 20, "hidden": 512, "num_layers": 3, "fc": [256, 128]},
                     num_classes=24, modality="both")
    g = torch.randn(8, 400, 40)
    s = torch.randn(8, 1950, 20)
    y = m(g, s)
    print("输出:", y.shape)
    n = sum(p.numel() for p in m.parameters())
    print(f"参数量: {n:,}")
