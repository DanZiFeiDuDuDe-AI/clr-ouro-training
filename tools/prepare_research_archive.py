"""Organize project sources for GitHub without altering original experiments.

Credentials are removed from publication copies only. Large/binary artifacts are
indexed by SHA-256 for a deduplicated Release archive, never silently discarded.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,mmap,os,re,shutil,subprocess,sys,tarfile
from pathlib import Path

WORK=Path(__file__).resolve().parents[1]
REPO=WORK/'github_publish'/'clr-ouro-training'
STATE=WORK/'.local'/'archive_state';STATE.mkdir(parents=True,exist_ok=True)
STUDIO=Path('H:/项目/造物工作室')
SKIP={'.git','.local','.ssh','.claude','.agents','.reasonix','.zcode','venv','.venv','__pycache__','.cache','node_modules','.ipynb_checkpoints','github_publish','server_full_archive','archive_packages'}
TEXT={'.py','.md','.txt','.json','.jsonl','.csv','.tsv','.toml','.yaml','.yml','.sh','.bat','.ps1','.tex','.bib','.html','.xml','.cfg','.ini','.log','.out','.aux','.gitattributes','.gitignore'}
BINARY={'.pt','.pth','.ckpt','.safetensors','.bin','.npy','.npz','.pkl','.pickle','.h5','.hdf5','.onnx','.gguf','.zip','.gz','.tar','.7z'}
CREDENTIAL_NAMES={'.env','auth.json','credentials.json','id_rsa','id_ed25519','scnet_password.dpapi'}
TOKENIZER_FILES={'tokenizer.json','vocab.json','merges.txt','tokenizer_config.json','added_tokens.json','special_tokens_map.json','tokenizer.model'}

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def source_roots(include_remote):
 roots=[(WORK,'research/looplm'),(Path('H:/CLR'),'research/clr'),(Path('D:/CLR'),'research/migrations/d_clr'),(Path('D:/LoopLM'),'research/migrations/d_looplm'),(Path('D:/PLAA'),'research/plaa')]
 roots +=[(STUDIO/name,'research/studio/'+name) for name in ('clr_v22_stage','clr_v22_closure_stage','clr_v23_oa_stage','single_slot_audit_stage')]
 if include_remote:roots.append((WORK/'server_full_archive/2026-10-20/scnet','archive/server-scnet'))
 return [(p,d) for p,d in roots if p.exists()]

def walk(root):
 for folder,dirs,names in os.walk(root,followlinks=False):
  dirs[:]=[d for d in sorted(dirs) if d not in SKIP and not (Path(folder)/d).is_symlink()]
  for name in sorted(names):
   p=Path(folder)/name
   if p.is_symlink() or name in CREDENTIAL_NAMES or p.suffix.lower() in {'.pyc','.pyo','.dpapi','.pem','.key','.partial'}:continue
   if p.is_file():yield p

def text_bytes(p):
 if p.name in TOKENIZER_FILES:return None,None
 if p.stat().st_size>32<<20:return None,None
 if p.suffix.lower() not in TEXT and p.name not in {'.gitignore','.gitattributes','LICENSE','NOTICE'}:return None,None
 data=p.read_bytes()
 for encoding in ('utf-8-sig','gb18030'):
  try:return data.decode(encoding),data
  except UnicodeError:pass
 return None,None

def secret_name(name):
 if re.search(r'(?i)(?:_file|_path|_env|_prompt|_name|_signature)$',name):return False
 return bool(re.search(r'(?i)password|passwd|passphrase|api_?key|hf_token|access_token|auth_token|ssh_pass|(?:^|_)secret_key$|^secret$',name))

TOKEN_PATTERN=r'\b(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b'

def usable_secret(value):
 return isinstance(value,str) and bool(value.strip()) and value not in {'password','changeme','your_password','SSH password: ','SSH password:','token','<secret>','<password>','EMPTY','DUMMY','YOUR_API_KEY','YOUR_TOKEN','None','none','null','secret','...','not-needed'} and not re.fullmatch(r'[A-Z][A-Z0-9]*_[A-Z0-9_]+',value) and not any(x in value for x in ('\n','${','os.environ','getpass','REDACTED')) and not value.startswith(('http://','https://'))

def credential_nodes(tree):
 """Return only literals in authentication fields, never arbitrary constants."""
 nodes=[];references=set()
 for n in ast.walk(tree):
  if isinstance(n,ast.keyword) and n.arg and secret_name(n.arg):
   if isinstance(n.value,ast.Name):references.add(n.value.id)
   elif isinstance(n.value,ast.Constant) and usable_secret(n.value.value):nodes.append(n.value)
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='connect' and len(n.args)>=4:
   v=n.args[3]
   if isinstance(v,ast.Name):references.add(v.id)
   elif isinstance(v,ast.Constant) and usable_secret(v.value):nodes.append(v)
  if isinstance(n,ast.Dict):
   for key,value in zip(n.keys,n.values):
    if isinstance(key,ast.Constant) and isinstance(key.value,str) and secret_name(key.value):
     if isinstance(value,ast.Name):references.add(value.id)
     elif isinstance(value,ast.Constant) and usable_secret(value.value):nodes.append(value)
 for n in ast.walk(tree):
  if isinstance(n,(ast.Assign,ast.AnnAssign)) and isinstance(getattr(n,'value',None),ast.Constant):
   targets=n.targets if isinstance(n,ast.Assign) else [n.target]
   if usable_secret(n.value.value) and any(isinstance(t,ast.Name) and (secret_name(t.id) or t.id in references) for t in targets):nodes.append(n.value)
 return list({id(n):n for n in nodes}.values())

def collect_secrets(files):
 found=set()
 for p in files:
  text,_=text_bytes(p)
  if text is None:continue
  if p.suffix.lower()=='.py':
   try:
    found.update(n.value for n in credential_nodes(ast.parse(text)))
   except SyntaxError:pass
  if p.suffix.lower() in {'.md','.txt','.sh','.ps1','.bat','.toml','.yaml','.yml'}:
   for pattern in [r'''(?i)(?:password|passwd|ssh_pass|api_key|hf_token)\s*[=:]\s*["'`]([^\s"'`]+)["'`]''',r'''(?i)sshpass\s+-p\s+["']([^\s"']+)["']''',r'''(?i)\(root\s*/\s*([^\s)]+)\)''']:
    for match in re.finditer(pattern,text):
     if usable_secret(match.group(1)):found.add(match.group(1))
  found.update(m.group(0) for m in re.finditer(TOKEN_PATTERN,text))
 return sorted(found,key=len,reverse=True)

def sanitize(p,secrets):
 text,data=text_bytes(p)
 if text is None:return None,0
 before=text;count=0
 def mask_auth_lines(value):
  nonlocal count
  lines=[]
  for line in value.splitlines(keepends=True):
   if re.search(r'(?i)password|passwd|passphrase|\bssh\b|sshpass|api_?key|hf_token|access_token|auth_token|credential|server|密码|口令|密钥|凭据|登录|连接|服务器|主机|端口|\broot\s*/',line):
    for secret in secrets:
     if secret in line:count+=line.count(secret);line=line.replace(secret,'REDACTED_CONFIGURE_ENV')
   def remove_token(m):
    nonlocal count
    count+=1;return 'REDACTED_CONFIGURE_ENV'
   line=re.sub(TOKEN_PATTERN,remove_token,line);lines.append(line)
  return ''.join(lines)
 if p.suffix.lower()=='.json':
  try:
   obj=json.loads(text)
   def visit(value):
    nonlocal count
    if isinstance(value,dict):
     return {key:('REDACTED_CONFIGURE_ENV' if secret_name(str(key)) and usable_secret(item) else visit(item)) for key,item in value.items()}
    if isinstance(value,list):return [visit(v) for v in value]
    if isinstance(value,str):return mask_auth_lines(value)
    return value
   edited=visit(obj)
   if edited==obj:return data,0
   count=max(count,1);return (json.dumps(edited,ensure_ascii=False,indent=2)+'\n').encode('utf-8'),count
  except (ValueError,TypeError):return data,0
 if p.suffix.lower()=='.py':
  try:
   encoded=text.encode('utf-8'); starts=[0]
   for line in encoded.splitlines(keepends=True):starts.append(starts[-1]+len(line))
   edits=[]
   for n in credential_nodes(ast.parse(text)):
    a=starts[n.lineno-1]+n.col_offset;b=starts[n.end_lineno-1]+n.end_col_offset
    edits.append((a,b,b'__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", "")'))
   for a,b,value in sorted(edits,reverse=True):encoded=encoded[:a]+value+encoded[b:];count+=1
   if edits:text=encoded.decode('utf-8')
  except SyntaxError:pass
 text=mask_auth_lines(text)
 # Markdown tables use a separate password cell rather than a quoted assignment.
 text,n=re.subn(r'(?im)^(\|\s*(?:SSH\s*)?密码\s*\|)[^\r\n]*',r'\1 已移除；请使用环境变量或本地受保护配置 |',text);count+=n
 if text==before:return data,0
 if p.suffix.lower()=='.py':
  # Preserve historical syntax errors but never introduce a new one.
  try:ast.parse(before)
  except SyntaxError:pass
  else:ast.parse(text)
 return text.encode('utf-8'),count

def move_previous():
 marker=STATE/'github_original_revision.json'
 if marker.exists():return json.loads(marker.read_text())
 rev=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip()
 names=subprocess.run(['git','ls-files','-z'],cwd=REPO,capture_output=True,check=True).stdout.decode().split('\0')
 old=REPO/'archive/github-2026-10-20'
 for rel in filter(None,names):
  source=REPO/rel;dest=old/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  if source.exists():source.replace(dest)
 record={'git_sha':rev,'files':len(list(filter(None,names))),'archive_path':'archive/github-2026-10-20'}
 marker.write_text(json.dumps(record,indent=2)+'\n');return record

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--local-only',action='store_true');args=ap.parse_args()
 sys.stdout.reconfigure(encoding='utf-8',errors='replace')
 old=move_previous();roots=source_roots(not args.local_only)
 original_git=STATE/'original_github_tree'
 if not (original_git/'.snapshot_complete').exists():
  original_git.mkdir(exist_ok=True)
  with tarfile.open(STATE/'original_github.tar') as archive:
   for member in archive:
    if not (member.isfile() or member.isdir()):continue
    destination=(original_git/member.name).resolve()
    if not destination.is_relative_to(original_git.resolve()):raise ValueError('Unsafe original Git path')
    if member.isdir():destination.mkdir(parents=True,exist_ok=True)
    else:
     destination.parent.mkdir(parents=True,exist_ok=True)
     with archive.extractfile(member) as source,destination.open('wb') as target:shutil.copyfileobj(source,target)
  (original_git/'.snapshot_complete').write_text(old['git_sha'])
 sources=[(p,d+'/'+p.relative_to(root).as_posix()) for root,d in roots for p in walk(root)]
 sources +=[(p,old['archive_path']+'/'+p.relative_to(original_git).as_posix()) for p in walk(original_git) if p.name!='.snapshot_complete']
 # A browsable paper library in addition to the exact source snapshots.
 papers=[(Path('H:/CLR/03_paper'),'papers/clr'),(Path('D:/PLAA/merged'),'papers/plaa/merged'),(Path('D:/PLAA/paper1'),'papers/plaa/framework'),(Path('D:/PLAA/paper2'),'papers/plaa/multiplicative')]
 for root,dest in papers:
  if root.exists():sources +=[(p,dest+'/'+p.relative_to(root).as_posix()) for p in walk(root)]
 for p in original_git.glob('*.docx'):sources.append((p,'papers/reports/'+p.name))
 files=list(dict.fromkeys(p for p,_ in sources));secrets=collect_secrets(files)
 print(json.dumps({'source_files':len(sources),'unique_sources':len(files),'credential_values_detected':len(secrets)}),flush=True)
 # Private cache contains only hashes/file metadata, never credentials.
 cache_path=STATE/'hash_cache.json';cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
 scan_path=STATE/'credential_scan_cache.json';secret_signature=hashlib.sha256('\0'.join(secrets).encode()).hexdigest()
 prior=json.loads(scan_path.read_text()) if scan_path.exists() else {}
 scanned=set(prior.get('objects',[])) if prior.get('secret_signature')==secret_signature else set()
 records=[];blobs={};redactions=[];groups={};seen=set()
 for index,(source,dest) in enumerate(sources,1):
  if dest in seen:raise ValueError('Duplicate publication path: '+dest)
  seen.add(dest);st=source.stat();key=str(source.resolve());finger=[st.st_size,st.st_mtime_ns]
  entry=cache.get(key)
  if entry and entry['finger']==finger:original=entry['sha256']
  else:original=sha(source);cache[key]={'finger':finger,'sha256':original}
  data,changes=sanitize(source,secrets)
  published=hashlib.sha256(data).hexdigest() if data is not None else original
  size=len(data) if data is not None else st.st_size
  binary=source.suffix.lower() in BINARY or size>8<<20
  target=REPO/dest
  if changes:redactions.append({'path':dest,'replacements':changes,'source_sha256':original,'published_sha256':published})
  row={'path':dest,'bytes':size,'sha256':published,'source':str(source),'source_sha256':original,'redacted':bool(changes),'storage':'release' if binary else 'git'}
  if binary:
   if changes:
    sanitized=STATE/'sanitized_artifacts'/published;sanitized.parent.mkdir(exist_ok=True);sanitized.write_bytes(data);blobsource=sanitized
   else:blobsource=source
   # A blob is only scanned once; the same artifact can serve many preserved paths.
   if published not in blobs:
    if size and source.name not in TOKENIZER_FILES and published not in scanned:
     with blobsource.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
      if any(mm.find(value.encode('utf-8'))>=0 for value in secrets):raise RuntimeError('Credential found in binary artifact; not published: '+dest)
     scanned.add(published)
    blobs[published]={'sha256':published,'bytes':size,'local_source':str(blobsource),'paths':[]}
   blobs[published]['paths'].append(dest)
   # Existing GitHub binary artifacts remain available locally for packing until
   # the Release upload succeeds. New ones are restored by restore_artifacts.py.
  else:
   target.parent.mkdir(parents=True,exist_ok=True)
   if target.resolve()!=source.resolve():
    if data is not None:target.write_bytes(data)
    else:shutil.copy2(source,target)
   elif changes:target.write_bytes(data)
  records.append(row)
  group='/'.join(dest.split('/')[:2]);g=groups.setdefault(group,{'files':0,'bytes':0});g['files']+=1;g['bytes']+=size
  if index%100==0:
   cache_path.write_text(json.dumps(cache,ensure_ascii=False),encoding='utf-8')
   scan_path.write_text(json.dumps({'secret_signature':secret_signature,'objects':sorted(scanned)}),encoding='utf-8')
   print(json.dumps({'staged':index,'total':len(sources),'unique_release_blobs':len(blobs)}),flush=True)
 manifest_dir=REPO/'manifests';manifest_dir.mkdir(exist_ok=True)
 (manifest_dir/'files.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
 (manifest_dir/'redactions.json').write_text(json.dumps({'files':redactions,'hardcoded_authentication_credentials_published':0,'tokenizer_handling':'Tokenizer vocabularies are preserved byte-for-byte; incidental vocabulary strings are not authentication credentials.'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 summary={'date':'2026-10-20','repository':'DanZiFeiDuDuDe-AI/clr-ouro-training','original_github':old,'complete_remote_included':not args.local_only,'files':len(records),'git_files':sum(r['storage']=='git' for r in records),'release_files':sum(r['storage']=='release' for r in records),'unique_release_blobs':len(blobs),'unique_release_bytes':sum(b['bytes'] for b in blobs.values()),'logical_file_bytes':sum(r['bytes'] for r in records),'redacted_files':len(redactions),'groups':groups,'runtime_exclusions':sorted(SKIP),'legacy_server':json.loads((STATE/'legacy_server_availability.json').read_text()) if (STATE/'legacy_server_availability.json').exists() else None}
 (manifest_dir/'archive_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 (STATE/'release_blobs_private.json').write_text(json.dumps(list(blobs.values()),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 cache_path.write_text(json.dumps(cache,ensure_ascii=False),encoding='utf-8')
 (STATE/'secret_fingerprints.json').write_text(json.dumps({'sha256':[hashlib.sha256(s.encode()).hexdigest() for s in secrets]}),encoding='utf-8')
 print(json.dumps({k:v for k,v in summary.items() if k not in {'groups','legacy_server','runtime_exclusions'}},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
