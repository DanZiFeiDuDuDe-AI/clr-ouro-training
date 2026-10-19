"""核对发布文件清单；可选同时核对已经恢复的大文件。"""
import argparse,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--with-artifacts',action='store_true');a=p.parse_args()
 records=[json.loads(line) for line in (ROOT/'manifests/files.jsonl').read_text(encoding='utf-8').splitlines() if line]
 checked=0;external=0
 for row in records:
  target=(ROOT/row['path']).resolve()
  if not target.is_relative_to(ROOT):raise ValueError('不安全的清单路径')
  if row['storage']=='release' and not a.with_artifacts:external+=1;continue
  if not target.is_file() or target.stat().st_size!=row['bytes'] or sha(target)!=row['sha256']:
   raise IOError('文件不存在或校验失败：'+row['path'])
  checked+=1
 print(json.dumps({'verified_files':checked,'release_files_not_requested':external,'all_requested_checks_passed':True}))

if __name__=='__main__':main()
