"""Read-only, resumable archive of this project's remote roots; SHA-256 verified."""
from __future__ import annotations
import base64, hashlib, json, os, shlex, shutil, sys, time
from collections import defaultdict
from pathlib import Path
from remote_weights import connect, execute, digest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'server_full_archive'/'2026-10-20'/'scnet'
SKIP={'.git','.local','.ssh','venv','.venv','__pycache__','.cache','node_modules','.ipynb_checkpoints'}

def inventory():
 code="""import os,json,stat,hashlib
from pathlib import Path
root=Path('/root/private_data'); files=[]; skipped=[]
skip={'.git','.local','.ssh','venv','.venv','__pycache__','.cache','node_modules','.ipynb_checkpoints'}
for project in sorted(root.iterdir()):
 if not project.is_dir() or not project.name.startswith('clr_'): continue
 for folder,dirs,names in os.walk(project,followlinks=False):
  for name in list(dirs):
   p=Path(folder)/name
   if name in skip or p.is_symlink():
    skipped.append({'path':str(p),'reason':'runtime/cache/credential/symlink'});dirs.remove(name)
  for name in sorted(names):
   p=Path(folder)/name
   if p.is_symlink() or not p.is_file():
    skipped.append({'path':str(p),'reason':'not a regular file'});continue
   h=hashlib.sha256()
   with p.open('rb') as f:
    for block in iter(lambda:f.read(8<<20),b''):h.update(block)
   s=p.stat();files.append({'path':str(p.relative_to(root)),'bytes':s.st_size,'mtime':s.st_mtime,'sha256':h.hexdigest()})
print(json.dumps({'source_root':str(root),'files':files,'excluded':skipped},ensure_ascii=True))
"""
 encoded=base64.b64encode(code.encode()).decode()
 result=execute('python -c '+shlex.quote('import base64;exec(base64.b64decode('+repr(encoded)+'))'),timeout=600)
 return json.loads(result)

def main():
 sys.stdout.reconfigure(encoding='utf-8',errors='replace');OUT.mkdir(parents=True,exist_ok=True)
 manifest_path=OUT.parent/'scnet_inventory.json'
 if manifest_path.exists(): manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
 else:
  print('HASHING_REMOTE_PROJECT_FILES',flush=True);manifest=inventory()
  manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'remote_files':len(manifest['files']),'remote_bytes':sum(r['bytes'] for r in manifest['files'])}),flush=True)
 # Reuse only files whose bytes match the remote digest; originals remain unchanged.
 candidates=defaultdict(list)
 for folder,dirs,names in os.walk(ROOT):
  dirs[:]=[d for d in dirs if d not in SKIP|{'github_publish','server_full_archive','archive_packages'}]
  for name in names:
   p=Path(folder)/name
   try:candidates[(name,p.stat().st_size)].append(p)
   except OSError:pass
 hashes={};c=None;sftp=None;records=[];downloaded=0;reused=0
 try:
  for index,row in enumerate(manifest['files'],1):
   rel=Path(row['path']); target=(OUT/rel).resolve()
   if not target.is_relative_to(OUT.resolve()):raise ValueError('Unsafe archive path')
   target.parent.mkdir(parents=True,exist_ok=True)
   expected=row['sha256']; method='existing'
   if target.exists():
    if digest(target)!=expected:raise FileExistsError('Different archive artifact preserved: '+str(target))
   else:
    match=None
    for candidate in candidates[(rel.name,row['bytes'])]:
     if str(candidate) not in hashes:hashes[str(candidate)]=digest(candidate)
     actual=hashes[str(candidate)]
     if actual==expected:match=candidate;break
    if match:
     shutil.copy2(match,target);method='verified_local_copy';reused+=row['bytes']
    else:
     remote=manifest['source_root']+'/'+row['path'];partial=target.with_name(target.name+'.partial')
     for attempt in range(4):
      try:
       if c is None:c=connect();sftp=c.open_sftp()
       # Resume large transfers at an already-downloaded byte offset.
       offset=partial.stat().st_size if partial.exists() else 0
       if offset>row['bytes']:partial.unlink();offset=0
       with sftp.open(remote,'rb') as source,partial.open('ab' if offset else 'wb') as dest:
        source.seek(offset)
        source.prefetch(file_size=row['bytes'],max_concurrent_requests=32)
        while True:
         block=source.read(2<<20)
         if not block:break
         dest.write(block)
       if digest(partial)!=expected:partial.unlink();raise IOError('Downloaded SHA-256 mismatch')
       partial.replace(target);downloaded+=row['bytes'];method='sftp';break
      except (OSError,EOFError) as error:
       if sftp:
        try:sftp.close()
        except Exception:pass
       if c:
        try:c.close()
        except Exception:pass
       c=sftp=None
       if attempt==3:raise
       print(json.dumps({'retry':index,'attempt':attempt+2,'error':type(error).__name__}),flush=True);time.sleep(2)
   if digest(target)!=expected:raise IOError('Final SHA-256 mismatch: '+str(target))
   records.append({**row,'method':method})
   if index%50==0 or row['bytes']>50_000_000:
    print(json.dumps({'verified':index,'total':len(manifest['files']),'downloaded_bytes':downloaded,'reused_bytes':reused,'path':row['path']}),flush=True)
 finally:
  if sftp:sftp.close()
  if c:c.close()
 (OUT.parent/'scnet_download_verification.json').write_text(json.dumps({'files':records,'downloaded_bytes':downloaded,'reused_bytes':reused,'complete':len(records)==len(manifest['files'])},indent=2)+'\n',encoding='utf-8')
 print('REMOTE_ARCHIVE_COMPLETE',flush=True)

if __name__=='__main__':main()
