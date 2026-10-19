"""Run one training command, then reconnect and verify its local artifacts."""
import argparse, json, time
from pathlib import Path
from remote_weights import execute, pull

def run(command,remote,local,timeout=3600):
    output=execute(command,timeout=timeout)
    # execute closes the training SSH connection; pull makes a new connection.
    for attempt in range(3):
        try:
            manifest=pull(remote,local)
            break
        except (OSError, EOFError) as error:
            if attempt==2: raise
            print(f'Download interrupted; reconnecting (attempt {attempt+2}/3, {type(error).__name__})',flush=True)
            time.sleep(2*(attempt+1))
    if not manifest['weights']: raise RuntimeError('Training returned without a checkpoint to download')
    print(output[-3000:])
    print(json.dumps({'training_complete':True,'weights_downloaded':manifest['weights'],'verified_files':len(manifest['verified_files']),'local':str(local)}),flush=True)
    return manifest

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('command'); ap.add_argument('remote'); ap.add_argument('local',type=Path); ap.add_argument('--timeout',type=int,default=3600)
    a=ap.parse_args(); run(a.command,a.remote,a.local,a.timeout)
