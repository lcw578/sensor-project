# -*- coding: utf-8 -*-
"""把 LaTeX 草稿转成纯文本，供 Light-Skills 的一致性/文风检查脚本消费。"""
import re
import sys
import pathlib

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "main_en.tex").read_text(encoding="utf-8")

# 表格与图替换为占位符，避免内容混入贡献句抽取
s = re.sub(r"\\begin\{table\}.*?\\end\{table\}", " [TABLE] ", src, flags=re.S)
s = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", " [FIGURE] ", s, flags=re.S)
s = re.sub(r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
           lambda m: "\n## Abstract\n" + m.group(1) + "\n", s, flags=re.S)
# section -> markdown 标题
s = re.sub(r"\\section\*?\{([^}]*)\}", lambda m: "\n\n## " + m.group(1) + "\n", s)
s = re.sub(r"\\subsection\{([^}]*)\}", lambda m: "\n\n### " + m.group(1) + "\n", s)
# 其余命令剥掉
s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", s)
s = s.replace("{", " ").replace("}", " ")
s = re.sub(r"[ \t]+", " ", s)
s = re.sub(r"\n{3,}", "\n\n", s)

out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "_draft_en.txt")
out.write_text(s.strip(), encoding="utf-8")
print(f"wrote {out} ({len(s)} chars)")
