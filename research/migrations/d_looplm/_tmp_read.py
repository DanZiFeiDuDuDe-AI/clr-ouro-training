from pathlib import Path
p = Path(r'D:\LoopLM\remote_mirror\p0h2_v2_train.py')
s = p.read_text(encoding='utf-8')
print(len(s))
print(s[:200])
