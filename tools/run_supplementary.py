# -*- coding: utf-8 -*-
"""补做实验编排器（支持并行；每路独立日志）

用法:
    python tools/run_supplementary.py --stage valid            # 只跑验证复现
    python tools/run_supplementary.py --stage p1 --par 3
    python tools/run_supplementary.py --stage p2 --par 3
    python tools/run_supplementary.py --stage p3 --par 3
    python tools/run_supplementary.py --list
"""
import os
import sys
import time
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "configs", "config_remote.yaml")
LOG = os.path.join(ROOT, "outputs", "logs")
PY = sys.executable

# ---------------- 实验清单 ----------------
# stage: valid 验证复现 / p1 模态dropout / p2 滤波 / p3 多种子
JOBS = [
    # --- 验证：必须复现本地 98.61% ---
    dict(stage="valid", tag="remote_verify", epochs=100, seed=42),

    # --- P1 模态 dropout：训练时随机置零一路 ---
    dict(stage="p1", tag="p1_moddrop50", epochs=60, seed=42, modality_dropout=0.5),
    dict(stage="p1", tag="p1_moddrop30", epochs=60, seed=42, modality_dropout=0.3),

    # --- P2 滤波后重训（cache 用 filtered 变体）---
    dict(stage="p2", tag="p2_filter_both", epochs=60, seed=42, cache_dir="filtered"),
    dict(stage="p2", tag="p2_filter_semg", epochs=60, seed=42, cache_dir="filtered",
         modality="semg"),
    dict(stage="p2", tag="p2_base_semg", epochs=60, seed=42, modality="semg"),

    # --- P3 多随机种子：主模型 + TCN ---
    dict(stage="p3", tag="p3_bilstm_s0", epochs=60, seed=0),
    dict(stage="p3", tag="p3_bilstm_s1", epochs=60, seed=1),
    dict(stage="p3", tag="p3_bilstm_s2", epochs=60, seed=2),
    dict(stage="p3", tag="p3_tcn_s0", epochs=60, seed=0, model="tcn"),
    dict(stage="p3", tag="p3_tcn_s1", epochs=60, seed=1, model="tcn"),
    dict(stage="p3", tag="p3_tcn_s2", epochs=60, seed=2, model="tcn"),
]


def build_cmd(j):
    cmd = [PY, "-u", os.path.join(ROOT, "src", "train.py"), "--config", CFG,
           "--tag", j["tag"], "--modality", j.get("modality", "both"),
           "--model", j.get("model", "bilstm"), "--epochs", str(j.get("epochs", 60))]
    if j.get("seed") is not None:
        cmd += ["--seed", str(j["seed"])]
    if j.get("modality_dropout"):
        cmd += ["--modality-dropout", str(j["modality_dropout"])]
    if j.get("cache_dir"):
        cmd += ["--cache-dir", os.path.join(ROOT, "outputs", "cache", j["cache_dir"])]
    return cmd


def run_pool(jobs, par):
    os.makedirs(LOG, exist_ok=True)
    RES = os.path.join(ROOT, "outputs", "results")
    # 断点续跑：已产出 summary 的实验直接跳过
    todo, skipped = [], []
    for j in jobs:
        if os.path.exists(os.path.join(RES, f"{j['tag']}_summary.json")):
            skipped.append(j["tag"])
        else:
            todo.append(j)
    if skipped:
        print(f"跳过已完成的 {len(skipped)} 个: {', '.join(skipped)}")
    if not todo:
        print("没有需要执行的实验")
        return []
    print(f"待执行 {len(todo)} 个\n")

    pending = list(todo)
    running = []          # (proc, job, logfile, t0)
    done = []
    while pending or running:
        while pending and len(running) < par:
            j = pending.pop(0)
            lf = open(os.path.join(LOG, f"{j['tag']}.log"), "w", encoding="utf-8")
            p = subprocess.Popen(build_cmd(j), stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT)
            print(f"[{time.strftime('%H:%M:%S')}] START {j['tag']:20s} pid={p.pid}", flush=True)
            running.append((p, j, lf, time.time()))
        time.sleep(5)
        for item in running[:]:
            p, j, lf, t0 = item
            if p.poll() is not None:
                lf.close()
                dt = time.time() - t0
                done.append((j["tag"], p.returncode, dt))
                print(f"[{time.strftime('%H:%M:%S')}] DONE  {j['tag']:20s} "
                      f"rc={p.returncode} {dt/60:.1f}min", flush=True)
                running.remove(item)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None, help="valid | p1 | p2 | p3 | all")
    ap.add_argument("--par", type=int, default=3, help="并行路数")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or not args.stage:
        for j in JOBS:
            print(f"  {j['stage']:6s} {j['tag']:20s} epochs={j.get('epochs')} "
                  f"seed={j.get('seed')} mod={j.get('modality','both')} "
                  f"model={j.get('model','bilstm')} "
                  f"dropout={j.get('modality_dropout',0)} cache={j.get('cache_dir','base')}")
        return

    jobs = JOBS if args.stage == "all" else [j for j in JOBS if j["stage"] == args.stage]
    if not jobs:
        sys.exit(f"没有匹配 stage={args.stage} 的实验")
    print(f"将执行 {len(jobs)} 个实验，并行度 {args.par}\n")
    t0 = time.time()
    done = run_pool(jobs, args.par)
    print(f"\n全部完成，总耗时 {(time.time()-t0)/60:.1f} min")
    for tag, rc, dt in done:
        print(f"  {'OK ' if rc==0 else 'FAIL'} {tag:20s} {dt/60:5.1f}min rc={rc}")


if __name__ == "__main__":
    main()
