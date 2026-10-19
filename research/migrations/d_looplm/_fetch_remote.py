import paramiko, os
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('cheems', username='ai training', password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=15, banner_timeout=15, auth_timeout=15)
sftp=c.open_sftp()
files = {
 'p0h2_v2_train.py':'D:/training/p0h2_v2_train.py',
 'p0h2_v2_cache.py':'D:/training/p0h2_v2_cache.py',
 'p0h2_lora.py':'D:/training/p0h2_lora.py',
 'verify_lora_norm1800.py':'D:/training/verify_lora_norm1800.py',
 'matched_eval_n150.py':'D:/training/matched_eval_n150.py',
 'eval_h2_step348_p0h5.py':'D:/training/eval_h2_step348_p0h5.py',
}
os.makedirs(r'D:\LoopLM\remote_mirror', exist_ok=True)
for local,remote in files.items():
    try:
        sftp.get(remote, r'D:\LoopLM\remote_mirror\\'+local)
        print('fetched',local)
    except Exception as e:
        print('FAIL',local,e)
for f in ['train_v2.log','train_v2.err.log','cache_v2.log','cache_v2.err.log','monitor_history.json','verify_lora_result.json']:
    try:
        sftp.get('D:/training/checkpoints/ouro_retention_m2_h2v2/'+f, r'D:\LoopLM\remote_mirror\\h2v2_'+f)
        print('fetched',f)
    except Exception as e:
        print('FAIL',f,e)
sftp.close(); c.close()
