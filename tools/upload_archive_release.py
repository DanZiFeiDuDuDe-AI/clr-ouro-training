"""Upload and verify the prepared archive Release, preserving completed assets."""
from __future__ import annotations
import hashlib,json,subprocess,sys,time
from pathlib import Path

WORK=Path(__file__).resolve().parents[1]
REPO='DanZiFeiDuDuDe-AI/clr-ouro-training';TAG='archive-2026-10-20'
FILES=WORK/'archive_packages/2026-10-20'
LOCAL=WORK/'github_publish/clr-ouro-training'
STATE=WORK/'.local/archive_state'

def command(args):
 r=subprocess.run(args,capture_output=True,text=True,encoding='utf-8',errors='replace')
 if r.returncode:raise RuntimeError('Command failed: '+' '.join(args[:4])+'\n'+r.stderr[-2000:])
 return r.stdout

def api(path):return json.loads(command(['gh','api',path]))

def release():
 matches=[r for r in api('repos/'+REPO+'/releases?per_page=100') if r['tag_name']==TAG]
 return matches[0] if matches else None

def get_assets(r):return api('repos/'+REPO+'/releases/'+str(r['id'])+'/assets?per_page=100')

def metadata_matches(asset,expected):
 if asset['size']!=expected['bytes'] or asset.get('state')!='uploaded':return False
 digest=asset.get('digest')
 return digest in (None,'sha256:'+expected['sha256'])

def main():
 sys.stdout.reconfigure(encoding='utf-8',errors='replace')
 manifest=json.loads((FILES/'artifacts.json').read_text(encoding='utf-8'))
 commit=json.loads((STATE/'publication_commit.json').read_text())['sha']
 remote=api('repos/'+REPO+'/git/ref/heads/main')['object']['sha']
 if remote!=commit:raise RuntimeError('The prepared code commit is not the remote main head')
 assets=[dict(p) for p in manifest['parts']]
 payload=FILES/'artifacts.json'
 assets.append({'name':payload.name,'bytes':payload.stat().st_size,'sha256':hashlib.sha256(payload.read_bytes()).hexdigest()})
 expected={p['name']:p for p in assets}
 notes=STATE/'release_notes_zh.md'
 notes.write_text('\n'.join([
  '# CLR / Ouro 研究资料归档', '',
  '整理日期：2026-10-20。代码提交：`'+commit+'`。', '',
  '- 汇总 3,584 条来源文件记录，代码、论文、报告和来源清单已推送到仓库。',
  '- 当前 SCNet 服务器 763 个研究文件已完整拉取并验证 SHA-256。',
  '- 权重、adapter、特征缓存和数据文件按 SHA-256 去重为 460 个对象，压缩为 18 个分卷，共 '+str(manifest['compressed_bytes'])+' 字节。',
  '- 所有分卷和全部 460 个对象均已在本地完整流式校验；tokenizer、模型权重和科学数据 JSON 保持来源字节。',
  '- 包括造物工作室的 v2.2 / v2.3 / single-slot 阶段资料，以及 CLR / PLAA 论文源码、PDF、参考文献和历史稿。',
  '- 旧 cheems 当前不可达；已有本地和原 GitHub 镜像已保留，不宣称认证旧机当前全部磁盘内容。',
  '- 未启动训练或 integrated O3 evaluation，O3-REBASE2 的历史 INCOMPLETE verdict 保持。', '',
  '## 恢复', '', '先克隆仓库，再执行：', '', '```bash',
  'python -m pip install zstandard','python tools/restore_artifacts.py','```','',
  '需要 Python 3.11+ 和 GitHub CLI。分卷是同一条 tar.zst 流，不能分别解压；恢复工具将校验分卷和对象，并拒绝覆盖内容不同的已有权重。', '',
  '路径、大小和 SHA-256 见附件 `artifacts.json` 与仓库 `manifests/`。论文入口见仓库 `papers/README.md`。', ''
 ]),encoding='utf-8')
 r=release()
 if r is None:
  command(['gh','release','create',TAG,'--repo',REPO,'--draft','--target',commit,'--title','CLR / Ouro 研究资料归档 · 2026-10-20','--notes-file',str(notes)])
  r=release()
 if not r:raise RuntimeError('Cannot retrieve created draft Release')
 if not r['draft']:raise RuntimeError('Existing published Release is preserved; refusing to replace it')
 print(json.dumps({'draft_release_id':r['id'],'expected_assets':len(assets),'target_commit':commit}),flush=True)
 # A small first asset confirms digest support before the large batch.
 current={x['name']:x for x in get_assets(r)}
 if 'artifacts.json' not in current:command(['gh','release','upload',TAG,str(payload),'--repo',REPO])
 small={x['name']:x for x in get_assets(r)}['artifacts.json']
 if not metadata_matches(small,expected['artifacts.json']):raise RuntimeError('Manifest asset verification failed')
 print(json.dumps({'manifest_asset_verified':True,'github_digest_available':bool(small.get('digest'))}),flush=True)
 for attempt in range(1,5):
  current={x['name']:x for x in get_assets(r)};pending=[]
  for name,p in expected.items():
   existing=current.get(name)
   if existing and metadata_matches(existing,p):continue
   if existing:
    if existing.get('state')=='uploaded' and existing.get('size',0)>0:raise RuntimeError('Different existing asset preserved: '+name)
    command(['gh','release','delete-asset',TAG,name,'--repo',REPO,'--yes'])
   pending.append(FILES/name)
  if not pending:break
  print(json.dumps({'upload_batch_attempt':attempt,'pending_assets':len(pending),'pending_bytes':sum(p.stat().st_size for p in pending)}),flush=True)
  # gh handles independent assets in its supported batched upload workflow.
  result=subprocess.run(['gh','release','upload',TAG,*map(str,pending),'--repo',REPO],capture_output=True,text=True,encoding='utf-8',errors='replace')
  if result.returncode:
   print(json.dumps({'batch_return_code':result.returncode,'error':result.stderr[-1200:]}),flush=True)
   if attempt==4:raise RuntimeError('Release upload failed after bounded retries')
   time.sleep(3)
 final={x['name']:x for x in get_assets(r)};verified=[]
 for name,p in expected.items():
  asset=final.get(name)
  if not asset or not metadata_matches(asset,p):raise RuntimeError('Missing or mismatched asset: '+name)
  digest=asset.get('digest')
  if not digest:
   # Older API responses require an authenticated byte-stream verification.
   process=subprocess.Popen(['gh','api','repos/'+REPO+'/releases/assets/'+str(asset['id']),'-H','Accept: application/octet-stream'],stdout=subprocess.PIPE)
   h=hashlib.sha256()
   for block in iter(lambda:process.stdout.read(8<<20),b''):h.update(block)
   if process.wait() or h.hexdigest()!=p['sha256']:raise RuntimeError('Downloaded asset SHA-256 mismatch: '+name)
   digest='sha256:'+h.hexdigest()
  if digest!='sha256:'+p['sha256']:raise RuntimeError('GitHub digest mismatch: '+name)
  verified.append({'name':name,'bytes':p['bytes'],'sha256':p['sha256'],'github_digest':digest,'asset_id':asset['id']})
 (STATE/'github_release_asset_verification.json').write_text(json.dumps({'tag':TAG,'repository':REPO,'commit':commit,'assets':verified,'all_verified':True},indent=2)+'\n')
 command(['gh','release','edit',TAG,'--repo',REPO,'--draft=false','--latest','--notes-file',str(notes)])
 r=release()
 if r['draft']:raise RuntimeError('Release remains draft')
 command(['gh','repo','edit',REPO,'--description','CLR / Ouro 持续隐状态调制研究：代码、论文、实验报告与可校验模型产物'])
 result={'published':True,'release_url':r['html_url'],'repository_url':'https://github.com/'+REPO,'assets_verified':len(verified),'archive_parts':len(manifest['parts']),'commit':commit}
 (STATE/'github_publication_result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
