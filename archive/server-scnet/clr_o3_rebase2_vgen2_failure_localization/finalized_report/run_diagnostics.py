"""Local orchestrator: each training task must download before the next starts."""
import json, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'tools'))
from remote_weights import execute,pull
from train_and_pull import run
REMOTE='/root/private_data/clr_o3_rebase2_vgen2_failure_localization'
ENV='LD_LIBRARY_PATH=/opt/dtk-26.04/hip/lib:/opt/dtk-26.04/lib:/opt/dtk-26.04/.hyhal/rocm_smi/lib '

def task(name,args):
 local=ROOT/name
 if (local/'COMPLETE.json').exists() and (local/'download_verification.json').exists():
  print('ALREADY_VERIFIED '+name,flush=True); return
 # Recovery from a lost connection uses completed remote artifacts, never retrains silently.
 status=execute('test -f '+REMOTE+'/'+name+'/COMPLETE.json && echo COMPLETE || echo PENDING').strip()
 if status=='COMPLETE':
  manifest=pull(REMOTE+'/'+name,local)
  if not manifest['weights']: raise RuntimeError('Missing trained checkpoint')
  print('RECOVERED_AND_VERIFIED '+name,flush=True)
 else:
  print('START '+name,flush=True)
  run(ENV+'python '+REMOTE+'/diagnose.py '+args,REMOTE+'/'+name,local,timeout=600)

def main():
 # Training representations stay server-side, SHA-manifested; frozen VAL is mirrored locally.
 pull(REMOTE+'/prepare',ROOT/'prepare',exclude_names={'3k_train.npz','10k_train.npz'})
 print('PREPARE_DOWNLOADED',flush=True)
 tasks=[('3k','R1','mlp'),('3k','R1','linear'),('3k','R1','centroid'),('3k','R0','mlp'),('3k','R2','mlp'),('3k','R3','mlp'),('3k','R4','mlp'),('10k','R3','mlp')]
 for snapshot,stage,kind in tasks:
  task(f'probe_{snapshot}_{stage}_{kind}',f'probe --snapshot {snapshot} --stage {stage} --kind {kind}')
 raw=[json.loads((ROOT/f'probe_3k_R1_{kind}/metrics.json').read_text()) for kind in ('mlp','linear')]
 if any(r['metrics']['single_write']['accuracy']>=.95 and r['metrics']['UPDATE_latest']['accuracy']>=.95 for r in raw):
  (ROOT/'optimization_stop.json').unlink(missing_ok=True)
  for size,scale in [(256,.25),(256,1),(256,4),(1024,1)]:
   task(f'sanity_{size}_{scale:g}',f'sanity --size {size} --scale {scale}')
 else:
  (ROOT/'optimization_stop.json').write_text(json.dumps({'status':'NOT_RUN','reason':'no raw supervised probe cleared >=95% diagnostic gate; negative probes do not prove information-theoretic insufficiency'},indent=2)+'\n')
 print('ALL_DIAGNOSTICS_AND_DOWNLOADS_COMPLETE',flush=True)

if __name__=='__main__': main()
