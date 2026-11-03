# -*- coding: utf-8 -*-
"""通用工具：随机种子、指标、绘图。"""
import os
import json
import numpy as np
import torch


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer="cuda"):
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, modality="both"):
    """返回 (y_true, y_pred, y_prob, mean_loss)。"""
    import torch.nn as nn
    model.eval()
    crit = nn.CrossEntropyLoss()
    ys, ps, probs, losses = [], [], [], []
    for glove, semg, y in loader:
        glove, semg, y = glove.to(device), semg.to(device), y.to(device)
        logits = model(glove, semg)
        loss = crit(logits, y)
        prob = torch.softmax(logits, dim=1)
        losses.append(loss.item() * len(y))
        ys.append(y.cpu().numpy())
        probs.append(prob.cpu().numpy())
        ps.append(logits.argmax(1).cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    y_prob = np.concatenate(probs)
    return y_true, y_pred, y_prob, sum(losses) / len(y_true)


def compute_metrics(y_true, y_pred, num_classes=24):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix)
    acc = accuracy_score(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(num_classes))),
    }


def plot_curves(history, out_path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, history["train_loss"], label="Train Loss")
    ax[0].plot(ep, history["val_loss"], label="Val Loss")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title("Loss")
    ax[1].plot(ep, history["train_acc"], label="Train Acc")
    ax[1].plot(ep, history["val_acc"], label="Val Acc")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[1].set_title("Accuracy")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, class_names, out_path, normalize=True, title="Confusion Matrix"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cmn, annot=True, fmt=".2f" if normalize else "d",
                cmap="Reds", cbar=True, square=True, ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 7})
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.xticks(rotation=90); plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
