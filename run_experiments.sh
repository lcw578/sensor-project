#!/usr/bin/env bash
# 串行执行全部实验（GPU 只有一个，必须排队）
# 用法: bash run_experiments.sh [stage...]   例: bash run_experiments.sh modality model
# 可选环境变量: PYTHON=python  （指定训练用的 Python 解释器，默认 python）
set -u
cd "$(dirname "$0")" || exit 1
PY="${PYTHON:-python}"
mkdir -p outputs/logs

STAGES="${*:-modality model ablation sweep}"

run() {                      # run <logname> <cmd...>
  local name="$1"; shift
  echo "##########  [$name]  $(date '+%F %T')  ##########"
  "$@" 2>&1 | tee "outputs/logs/${name}.log"
  echo "##########  [$name] done  $(date '+%F %T')  ##########"
}

for s in $STAGES; do
  case "$s" in
    modality)  run modality  "$PY" src/experiments/modality_compare.py --epochs 100 ;;
    model)     run model     "$PY" src/experiments/model_compare.py --epochs 60 ;;
    ablation)  run ablation  "$PY" src/experiments/ablation.py --epochs 60 ;;
    sweep)     run sweep     "$PY" src/experiments/param_sweep.py --epochs 40 ;;
    *) echo "未知阶段: $s" ;;
  esac
done
echo "全部实验完成: $(date '+%F %T')"
