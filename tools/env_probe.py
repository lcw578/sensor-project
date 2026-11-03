# -*- coding: utf-8 -*-
"""环境探测：Python / PyTorch / CUDA / 算力基准（多环境通用同一脚本，便于对比）"""
import sys
import time
import shutil
import platform

print("=" * 60)
print("host:", platform.node(), "| python:", sys.version.split()[0], "|", platform.machine())
print("=" * 60)

try:
    import torch
    print(f"torch: {torch.__version__}  (cuda build: {torch.version.cuda})")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
        print(f"device_count: {torch.cuda.device_count()}  capability: {torch.cuda.get_device_capability(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_memory/1024**3:.1f} GB")
        # 算力基准：4000x4000 fp32 matmul x20
        x = torch.randn(4000, 4000, device="cuda")
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(20):
            x = (x @ x.T) * 1e-3
        torch.cuda.synchronize()
        dt = time.time() - t
        flops = 20 * 2 * 4000**3
        print(f"bench matmul 4000^3 x20: {dt:.2f}s  ->  {flops/dt/1e12:.1f} TFLOPS (fp32)")
        print(f"peak mem alloc: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
        # fp16 基准
        xh = x.half()
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(20):
            xh = (xh @ xh.T) * 1e-3
        torch.cuda.synchronize()
        dt = time.time() - t
        print(f"bench matmul fp16:      {dt:.2f}s  ->  {flops/dt/1e12:.1f} TFLOPS (fp16)")
        del x, xh
        torch.cuda.empty_cache()
except Exception as e:
    print("torch FAILED:", type(e).__name__, e)

print("--- 依赖 ---")
for mod in ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "PIL"]:
    try:
        m = __import__(mod)
        print(f"  {mod:12s} {getattr(m, '__version__', '?')}")
    except Exception:
        print(f"  {mod:12s} MISSING")

print("--- 磁盘 ---")
for d in ["C:/", "D:/"]:
    try:
        u = shutil.disk_usage(d)
        print(f"  {d} free {u.free/1024**3:6.1f} GB / total {u.total/1024**3:6.1f} GB")
    except Exception as e:
        print(f"  {d} {e}")
