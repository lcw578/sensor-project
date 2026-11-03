# 24 类多模态手语识别

基于柔性传感手套（左右手应变/压力）与表面肌电臂带（左右臂 sEMG）的双模态信号，对 24 类中国手语做分类识别。从零构建完整深度学习流水线：数据审计 → 防泄漏预处理 → 固定分层划分 → 双分支 Bi-LSTM 融合主模型 → 五组系统性实验（参数调节 / 模型对比 / 多模态对比 / 消融 / 融合机制）。

对标源论文：Hu et al., *Advanced Science* 2026, 13:e20586（3D-Printed Ion-Conductive Hydrogels ... for Multimodal Sign Language Recognition）。

## 结果速览

| 模型 | 测试准确率（288 样本） | 说明 |
|---|---|---|
| 双分支 Bi-LSTM 融合（主模型） | 98.61%（单次）/ 98.96±0.40%（4 种子） | 复现源论文设置 |
| TCN | 99.42±0.40%（3 种子），单次最高 99.65% | 参数量仅 0.30M（Bi-LSTM 的 1/50） |
| 仅应变/压力（手套） | 98.26% | 单模态已接近天花板 |
| 仅 sEMG | 91.32% | 滤波后降至 75.69% |

核心发现（详见 `reports/main.tex` 与 `docs/`）：推理剪枝显示去掉 sEMG 后 288/288 预测逐样本不变——**融合模型的 sEMG 分支在分类决策中完全未被使用**，所谓"多模态融合"功能上等价于纯手套单模态；Oracle 上界分析表明五组配置中四组已无理论改进空间。源论文"融合比单模态高 +15.9 pp"的核心论点在本数据集上不成立，属典型的主导模态（dominant modality）现象。

## 目录结构

```
configs/config.yaml      全局配置（路径为仓库相对路径，可用环境变量覆盖）
src/
  data/preprocess.py     阶段一：读取/拼接/划分/归一化，产出缓存
  data/filters.py        sEMG 带通/陷波滤波（可选）
  data/dataset.py        PyTorch Dataset / DataLoader
  models/bilstm.py       主模型：双分支 Bi-LSTM 融合
  models/others.py       对比模型：1D-CNN / Bi-GRU / TCN / Transformer
  train.py               阶段二：训练/评估主脚本
  experiments/           参数扫描 / 模型对比 / 多模态对比 / 消融
tools/                   数据体检、离群分析、泄漏审计、图表与报告辅助脚本
docs/                    方法决策、泄漏审计、补做实验等过程文档
reports/                 实验报告 LaTeX 源码（中文 main.tex / 英文 main_en.tex）与图表
deck/ deck_en/           幻灯片（HTML，Chrome 打开 deck.html 放映）
outputs/
  results/               各实验 summary（json/csv）
  figures/               曲线、混淆矩阵、对比图
  logs/                  训练日志
run_experiments.sh       串行跑完全部实验（单 GPU 排队）
```

## 环境要求

- Python 3.13，PyTorch 2.9.1+cu126（或其他 CUDA 12.6 版本；CPU 可跑但极慢）
- 依赖：`numpy pandas scipy scikit-learn matplotlib seaborn pyyaml`
- 主模型 batch=50 在 8 GB 显卡上需开启 AMP（配置已默认开启）

## 数据放置（数据集不入库）

仓库不含数据集。将数据集按如下结构放置到仓库根目录：

```
24手语数据/data/<类别名>/<1..120>.pkl     # 24 类 × 120 = 2880 个样本
```

每个 `.pkl` 为 dict，固定 4 键：`res1`/`res2`（左/右手应变压力，(20,400) float32，200 Hz）、`muscle1`/`muscle2`（左/右手 sEMG，(10,1950) float32，1000 Hz），单样本时长 2 s。

## 复现步骤

```bash
# 1. 预处理（读取 → 左右手通道拼接 → 8:1:1 分层划分 → 分位截断 → 逐通道标准化）
python src/data/preprocess.py

# 2. 训练主模型（默认：双分支 Bi-LSTM 融合，batch=50, lr=1e-3, dropout=0.3）
python src/train.py

# 3. 单模态对比 / 调参
python src/train.py --modality glove
python src/train.py --modality semg
python src/train.py --lr 0.0005 --batch-size 32 --tag lr5e-4

# 4. 批量实验（多模态对比 / 模型对比 / 消融 / 参数扫描，串行执行）
bash run_experiments.sh                      # 全部
bash run_experiments.sh modality model       # 指定阶段
```

划分以 `.pkl` 文件为单位、按类别分层（每类 96/12/12，总计 2304/288/288），固定种子 42，落盘 `outputs/cache/meta.csv`，全部实验复用同一划分；归一化统计量只从训练集计算。测试集仅在训练结束后评估一次。

## 配置

`configs/config.yaml` 中所有路径为仓库相对路径（启动时基于仓库根目录解析），可用环境变量覆盖为绝对路径：

| 环境变量 | 对应配置 |
|---|---|
| `SIGNLANG_DATA_ROOT` | `paths.data_root` |
| `SIGNLANG_CACHE_DIR` | `paths.cache_dir` |
| `SIGNLANG_CKPT_DIR` | `paths.ckpt_dir` |
| `SIGNLANG_FIG_DIR` | `paths.fig_dir` |
| `SIGNLANG_RES_DIR` | `paths.res_dir` |

## 说明

- 模型权重（`outputs/checkpoints/`）与预处理缓存默认不入库，重新训练即可生成。
- 第三方 PDF（源论文、教材、任务书）不入库。
- 测试集仅 288 样本，单次差异小于约 1 pp 不可分辨；报告中对模型间差异统一采用多种子统计口径。
