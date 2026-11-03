# -*- coding: utf-8 -*-
"""
对比模型库（用于实验要求 3「不同模型对比」）

统一接口：forward(glove, semg) -> logits (B, num_classes)
输入：glove (B, 400, 40)  semg (B, 1950, 20)   —— 与主模型完全一致，便于公平对比。

实现：CNN1D / GRU / TCN / Transformer 四种双分支模型 + 一个通用双分支容器。
所有分支最终输出固定维度特征 (out_dim=128)，拼接后经分类头输出。
"""
import math
import torch
import torch.nn as nn


# ------------------------------------------------------------------ 通用容器
class DualBranchModel(nn.Module):
    """把两个分支的输出拼接后分类；支持 modality ∈ {both, glove, semg}。"""

    def __init__(self, branch_g, branch_s, num_classes=24, dropout=0.3, modality="both"):
        super().__init__()
        assert modality in ("both", "glove", "semg")
        self.modality = modality
        self.branch_g = branch_g
        self.branch_s = branch_s
        dims = {"both": branch_g.out_dim + branch_s.out_dim,
                "glove": branch_g.out_dim, "semg": branch_s.out_dim}
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dims[modality], num_classes))

    def forward(self, glove, semg):
        feats = []
        if self.modality in ("both", "glove"):
            feats.append(self.branch_g(glove))
        if self.modality in ("both", "semg"):
            feats.append(self.branch_s(semg))
        z = feats[0] if len(feats) == 1 else torch.cat(feats, dim=1)
        return self.head(z)


# ------------------------------------------------------------------ CNN1D
class CNN1DBranch(nn.Module):
    """1D 卷积分支：逐层卷积 + 池化 + 全局平均池化。输入 (B,T,C)。"""

    def __init__(self, input_dim, channels=(64, 128, 128), kernels=(7, 5, 3),
                 dropout=0.3, out_dim=128):
        super().__init__()
        layers, cin = [], input_dim
        for i, (co, k) in enumerate(zip(channels, kernels)):
            layers += [nn.Conv1d(cin, co, k, padding=k // 2),
                       nn.BatchNorm1d(co), nn.ReLU(inplace=True),
                       nn.MaxPool1d(2)]
            cin = co
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(cin, out_dim),
                                  nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.out_dim = out_dim

    def forward(self, x):                       # (B,T,C)
        x = x.transpose(1, 2)                   # -> (B,C,T)
        h = self.conv(x)
        h = self.pool(h)
        return self.proj(h)


# ------------------------------------------------------------------ GRU
class GRUBranch(nn.Module):
    """把主模型的 Bi-LSTM 换成 Bi-GRU，其它结构保持一致。"""

    def __init__(self, input_dim, hidden=128, num_layers=3, dropout=0.3,
                 out_dim=128, pooling="last"):
        super().__init__()
        self.pooling = pooling
        self.rnn = nn.GRU(input_dim, hidden, num_layers=num_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.proj = nn.Sequential(nn.Linear(hidden * 2, out_dim), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout))
        self.out_dim = out_dim

    def forward(self, x):
        out, _ = self.rnn(x)
        h = out.mean(dim=1) if self.pooling == "mean" else out[:, -1, :]
        return self.proj(h)


# ------------------------------------------------------------------ TCN
class _TCNBlock(nn.Module):
    def __init__(self, cin, cout, k, dilation, dropout):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(cin, cout, k, dilation=dilation)
        self.conv2 = nn.Conv1d(cout, cout, k, dilation=dilation)
        self.bn1, self.bn2 = nn.BatchNorm1d(cout), nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):                       # (B,C,T)
        h = self.relu(self.bn1(self.conv1(nn.functional.pad(x, (self.pad, 0)))))
        h = self.drop(h)
        h = self.relu(self.bn2(self.conv2(nn.functional.pad(h, (self.pad, 0)))))
        return self.relu(h + self.down(x))


class TCNBranch(nn.Module):
    """时序卷积网络分支（膨胀因果卷积 + 残差）。输入 (B,T,C)。"""

    def __init__(self, input_dim, channels=(64, 64, 128), kernel=3,
                 dilations=(1, 2, 4), dropout=0.3, out_dim=128):
        super().__init__()
        blocks, cin = [], input_dim
        for co, d in zip(channels, dilations):
            blocks.append(_TCNBlock(cin, co, kernel, d, dropout))
            cin = co
        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(cin, out_dim),
                                  nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.out_dim = out_dim

    def forward(self, x):
        h = self.tcn(x.transpose(1, 2))
        return self.proj(self.pool(h))


# ------------------------------------------------------------------ Transformer
class _PosEnc(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerBranch(nn.Module):
    """下采样卷积降序列长度 -> 位置编码 -> TransformerEncoder -> 平均池化。"""

    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2,
                 dropout=0.3, out_dim=128, downsample=8):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=downsample, stride=downsample),
            nn.BatchNorm1d(d_model), nn.ReLU(inplace=True))
        self.pos = _PosEnc(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4,
                                         dropout=dropout, batch_first=True,
                                         norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.proj = nn.Sequential(nn.Linear(d_model, out_dim), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout))
        self.out_dim = out_dim

    def forward(self, x):
        h = self.down(x.transpose(1, 2)).transpose(1, 2)     # (B,T',d)
        h = self.enc(self.pos(h))
        return self.proj(h.mean(dim=1))


# ------------------------------------------------------------------ 工厂
_REGISTRY = {
    "cnn": lambda d, dim, dp: CNN1DBranch(dim, dropout=dp),
    "gru": lambda d, dim, dp: GRUBranch(dim, hidden=d.get("hidden", 128),
                                        num_layers=d.get("num_layers", 3), dropout=dp),
    "tcn": lambda d, dim, dp: TCNBranch(dim, dropout=dp),
    "transformer": lambda d, dim, dp: TransformerBranch(dim, dropout=dp),
}


def build_model_by_name(name, cfg, dropout=0.3, modality="both", **kw):
    """name ∈ {bilstm, cnn, gru, tcn, transformer}；cfg 为 config.yaml 的 model 段。
    kw 可透传 use_fc / fusion / bidirectional / pooling 等（用于消融实验）。
    """
    if name == "bilstm":
        from src.models.bilstm import BiLSTMFusion
        return BiLSTMFusion(cfg["glove"], cfg["semg"], num_classes=cfg.get("num_classes", 24),
                            dropout=dropout, modality=modality, **kw)
    key = name.lower().replace("1d", "").replace("-", "")
    if key not in _REGISTRY:
        raise ValueError(f"未知模型: {name}（可选 {['bilstm'] + list(_REGISTRY)}）")
    g = _REGISTRY[key](cfg.get("glove", {}), cfg["glove"]["input_dim"], dropout)
    s = _REGISTRY[key](cfg.get("semg", {}), cfg["semg"]["input_dim"], dropout)
    return DualBranchModel(g, s, num_classes=cfg.get("num_classes", 24),
                           dropout=dropout, modality=modality)


if __name__ == "__main__":
    cfg = {"glove": {"input_dim": 40}, "semg": {"input_dim": 20}, "num_classes": 24}
    g, s = torch.randn(8, 400, 40), torch.randn(8, 1950, 20)
    for nm in ["bilstm", "cnn", "gru", "tcn", "transformer"]:
        m = build_model_by_name(nm, cfg, modality="both")
        o = m(g, s)
        n = sum(p.numel() for p in m.parameters())
        print(f"{nm:12s} out={tuple(o.shape)}  params={n:,}")
