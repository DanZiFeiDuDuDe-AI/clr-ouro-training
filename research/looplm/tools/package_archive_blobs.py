"""Build a deduplicated, split tar.gz Release archive from verified artifacts."""
from __future__ import annotations
import hashlib,io,json,os,sys,tarfile,zstandard
from pathlib import Path

WORK=Path(__file__).resolve().parents[1]
REPO=WORK/'github_publish/clr-ouro-training'
OUT=WORK/'archive_packages'/'2026-10-20';OUT.mkdir(parents=True,exist_ok=True)
PREFIX='research-artifacts-2026-10-20.tar.zst.part-'

class SplitWriter:
 def __init__(self):self.file=None;self.size=0;self.count=0;self.total=0;self.parts=[];self.hasher=None
 def writable(self):return True
 def write(self,data):
  original=len(data)
  while data:
   if self.file is None:
    self.count+=1;self.path=OUT/(PREFIX+f'{self.count:04d}');self.file=self.path.open('wb');self.size=0;self.hasher=hashlib.sha256()
   part=data[:min(len(data),(512<<20)-self.size)];self.file.write(part);self.hasher.update(part);self.size+=len(part);self.total+=len(part);data=data[len(part):]
   if self.size==512<<20:self.finish_part()
  return original
 def flush(self):
  if self.file:self.file.flush()
 def finish_part(self):
  if self.file:
   self.file.close();self.parts.append({'name':self.path.name,'bytes':self.size,'sha256':self.hasher.hexdigest()});self.file=None
   print(json.dumps({'part':self.count,'bytes':self.size,'compressed_total':self.total}),flush=True)
 def close(self):self.finish_part()

def main():
 sys.stdout.reconfigure(encoding='utf-8')
 blobs=json.loads((WORK/'.local/archive_state/release_blobs_private.json').read_text(encoding='utf-8'))
 writer=SplitWriter()
 with zstandard.ZstdCompressor(level=3,threads=4,write_checksum=True).stream_writer(writer,closefd=False) as zipped:
  with tarfile.open(fileobj=zipped,mode='w|',format=tarfile.PAX_FORMAT) as archive:
   for index,blob in enumerate(sorted(blobs,key=lambda x:x['sha256']),1):
    path=Path(blob['local_source']);info=tarfile.TarInfo('objects/'+blob['sha256']);info.size=blob['bytes'];info.mtime=0;info.mode=0o644
    with path.open('rb') as f:archive.addfile(info,f)
    if blob['bytes']>100_000_000:print(json.dumps({'packed_object':index,'object_bytes':blob['bytes'],'objects':len(blobs)}),flush=True)
 writer.close()
 manifest={'format':'split-tar-zstd-v1','compression':{'codec':'zstd','version':zstandard.__version__,'level':3,'threads':4},'tag':'archive-2026-10-20','repository':'DanZiFeiDuDuDe-AI/clr-ouro-training','parts':writer.parts,'compressed_bytes':writer.total,'unique_blob_bytes':sum(b['bytes'] for b in blobs),'blobs':[{'sha256':b['sha256'],'bytes':b['bytes'],'paths':b['paths']} for b in sorted(blobs,key=lambda x:x['sha256'])]}
 for target in [OUT/'artifacts.json',REPO/'manifests/artifacts.json']:target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'archive_complete':True,'parts':len(writer.parts),'bytes':writer.total,'objects':len(blobs)}),flush=True)

if __name__=='__main__':main()
