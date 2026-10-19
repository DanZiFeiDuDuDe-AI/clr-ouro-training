import paramiko

cmd = r'''cd /d D:\training && py -3 -c "import json, numpy as np, pathlib, inspect, a1_ls2_cache, clr_v2; p=pathlib.Path(r'D:\training\checkpoints\clr_v24_o21a_mr\cache'); m=json.loads((p/'train_meta.jsonl').read_text(encoding='utf-8').splitlines()[0]); print(json.dumps({'files': sorted(x.name for x in p.iterdir()), 'keys': {s: sorted(np.load(p/(s+'_data.npz'), allow_pickle=False).files) for s in ['train','val','testA','testB']}, 'manifest': json.loads((p/'manifest.json').read_text(encoding='utf-8')), 'sample_meta': m, 'extract_h_batch': inspect.getsource(a1_ls2_cache.extract_h_batch), 'load_ouro': inspect.getsource(clr_v2.load_ouro)}, ensure_ascii=False, indent=2))"'''
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("100.104.57.18", username="ai training", password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=30)
_, o, e = c.exec_command(cmd)
print(o.read().decode("utf-8", "replace").encode("ascii", "replace").decode("ascii"), end="")
print(e.read().decode("utf-8", "replace").encode("ascii", "replace").decode("ascii"), end="")
c.close()
