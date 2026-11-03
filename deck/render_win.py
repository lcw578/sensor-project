# -*- coding: utf-8 -*-
"""逐页渲染 deck（Windows 异步 Chrome 兼容版），并拼 contact sheet。"""
import os
import sys
import time
import pathlib
import subprocess
import tempfile
from PIL import Image

CHROME = os.environ.get("CHROME", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
UDD = os.path.join(tempfile.gettempdir(), "chrome-headless-deck")

deck = pathlib.Path(sys.argv[1]).resolve()
n = int(sys.argv[2])
out = pathlib.Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
uri = deck.as_uri()

for i in range(1, n + 1):
    png = out / f"slide-{i:02d}.png"
    if png.exists() and png.stat().st_size > 10000:
        continue
    # 注意：--no-sandbox 会让本机 Chrome 的 headless 截图静默失败（实测），不能加
    win_png = str(png).replace("/", "\\")
    subprocess.Popen([CHROME, "--headless=new", "--disable-gpu",
                      "--hide-scrollbars", f"--user-data-dir={UDD}\\p{i}",
                      "--virtual-time-budget=6000", "--force-device-scale-factor=1",
                      f"--screenshot={win_png}", "--window-size=1920,1080",
                      uri + f"#{i}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # 等文件落盘（Chrome 会 de-elevated 重生成进程，父进程立即返回）
    t0 = time.time()
    while time.time() - t0 < 40:
        if png.exists() and png.stat().st_size > 10000:
            time.sleep(0.4)
            break
        time.sleep(0.3)
    print(png.name, png.exists() and png.stat().st_size, flush=True)

# contact sheet: 4 列
tiles = []
missing = []
for i in range(1, n + 1):
    p = out / f"slide-{i:02d}.png"
    if p.exists():
        tiles.append(Image.open(p))
    else:
        missing.append(i)
if missing:
    print("缺失页:", missing)
if tiles:
    tw, th, cols, gap = 480, 270, 4, 10
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * gap, rows * th + (rows + 1) * gap), (200, 205, 212))
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        sheet.paste(t.resize((tw, th), Image.LANCZOS), (gap + c * (tw + gap), gap + r * (th + gap)))
    sheet.save(out / "contact.png")
    print("contact", sheet.size, "->", out / "contact.png")
