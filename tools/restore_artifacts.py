"""从 GitHub Release 恢复权重和特征缓存；默认保留不同内容的已有文件。"""
from __future__ import annotations
import argparse,gzip,hashlib,io,json,re,shutil,subprocess,sys,tarfile,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def decompressed(stream,format):
 if format=='split-tar-zstd-v1':
  import zstandard
  return zstandard.ZstdDecompressor().stream_reader(stream)
 if format=='split-tar-gzip-v1':return gzip.GzipFile(fileobj=stream,mode='rb')
 raise ValueError('不支持的归档格式：'+format)

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

class PartsReader(io.RawIOBase):
 def __init__(self,paths,expected):
  self.paths=iter(zip(paths,expected));self.current=None;self.expected=None;self.hasher=None;self.count=0;self.verified=0
 def readable(self):return True
 def readinto(self,b):
  while True:
   if self.current is None:
    try:
     self.path,self.expected=next(self.paths);self.current=self.path.open('rb');self.hasher=hashlib.sha256();self.count=0
    except StopIteration:return 0
   n=self.current.readinto(b)
   if n:
    self.hasher.update(memoryview(b)[:n]);self.count+=n;return n
   self.current.close();self.current=None
   if self.count!=self.expected['bytes'] or self.hasher.hexdigest()!=self.expected['sha256']:
    preserved=self.path.with_name(self.path.name+'.invalid-'+str(time.time_ns()));self.path.replace(preserved)
    raise IOError('分卷校验失败，已保留坏卷：'+str(preserved))
   self.verified+=1
 def close(self):
  if self.current:self.current.close()
  super().close()

def main():
 if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='replace')
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--cache',type=Path,default=ROOT/'.archive-cache');ap.add_argument('--output',type=Path,default=ROOT)
 ap.add_argument('--verify-only',action='store_true',help='只校验归档，不写入权重')
 ap.add_argument('--offline',action='store_true',help='只使用 cache 中已有分卷')
 a=ap.parse_args();m=json.loads((ROOT/'manifests/artifacts.json').read_text(encoding='utf-8'))
 if m['format']=='split-tar-zstd-v1':
  try:import zstandard
  except ImportError:raise SystemExit('请先安装解压依赖：python -m pip install zstandard')
 a.cache.mkdir(parents=True,exist_ok=True);paths=[]
 for part in m['parts']:
  p=a.cache/part['name']
  valid=p.exists() and p.stat().st_size==part['bytes']
  if p.exists() and not valid:
   if a.offline:raise IOError('离线分卷校验失败：'+str(p))
   p.replace(p.with_name(p.name+'.invalid-'+str(time.time_ns())))
  if not p.exists():
   if a.offline:raise FileNotFoundError(p)
   subprocess.run(['gh','release','download',m['tag'],'--repo',m['repository'],'--pattern',part['name'],'--dir',str(a.cache)],check=True)
   valid=p.stat().st_size==part['bytes']
  if not valid:raise IOError('分卷校验失败：'+str(p))
  paths.append(p)
 print(f'已找到 {len(paths)} 个分卷，将在流式读取时验证 SHA-256',flush=True)
 wanted={b['sha256']:b for b in m['blobs']};found=set();restored=0
 output=a.output.resolve();objects=a.cache/'objects';objects.mkdir(exist_ok=True)
 with PartsReader(paths,m['parts']) as joined,io.BufferedReader(joined,buffer_size=8<<20) as buffered,decompressed(buffered,m['format']) as unzipped,tarfile.open(fileobj=unzipped,mode='r|') as archive:
  for member in archive:
   if not member.isfile() or not member.name.startswith('objects/'):raise ValueError('归档中有不支持的条目')
   sha=member.name.split('/')[-1]
   if not re.fullmatch('[a-f0-9]{64}',sha) or member.name!='objects/'+sha:raise ValueError('不安全的对象名称')
   if sha not in wanted or sha in found:raise ValueError('未知或重复的归档对象')
   blob=wanted[sha]
   if member.size!=blob['bytes']:raise IOError('对象大小不匹配')
   source=archive.extractfile(member);h=hashlib.sha256();temp=objects/(sha+'.partial');dest=None
   try:
    if not a.verify_only:dest=temp.open('wb')
    for block in iter(lambda:source.read(8<<20),b''):
     h.update(block)
     if dest:dest.write(block)
   finally:
    source.close()
    if dest:dest.close()
   if h.hexdigest()!=sha:raise IOError('对象校验失败：'+sha)
   found.add(sha)
   if not a.verify_only:
    for rel in blob['paths']:
     target=(output/rel).resolve()
     if not target.is_relative_to(output):raise ValueError('不安全的目标路径')
     if target.exists():
      if digest(target)!=sha:raise FileExistsError('已保留内容不同的本地文件：'+str(target))
      continue
     target.parent.mkdir(parents=True,exist_ok=True);pending=target.with_name(target.name+'.partial');shutil.copyfile(temp,pending);pending.replace(target);restored+=1
    temp.unlink()
   if len(found)%100==0:print(f'已校验对象 {len(found)}/{len(wanted)}',flush=True)
  # Consume the compression footer and any buffered final bytes, then close out
  # the final compressed part's hash. Every restored object was already hashed.
  while unzipped.read(8<<20):pass
  while buffered.read(8<<20):pass
  if joined.verified!=len(paths):raise IOError('分卷未全部完成校验')
 if found!=set(wanted):raise IOError('归档对象不完整')
 print(json.dumps({'verified_objects':len(found),'verified_parts':joined.verified,'restored_files':restored,'verify_only':a.verify_only}),flush=True)

if __name__=='__main__':main()
