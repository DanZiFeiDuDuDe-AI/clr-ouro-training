import paramiko, time
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('cheems', username='ai training', password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=15, banner_timeout=15, auth_timeout=15)
for cmd in ['dir /b D:\\training\\checkpoints\\ouro_retention_m2_h2v2_smoke', 'type D:\\training\\checkpoints\\ouro_retention_m2_h2v2_smoke\\monitor_history.json 2>nul', 'type D:\\training\\checkpoints\\ouro_retention_m2_h2v2_smoke\\train_log.json 2>nul', 'type D:\\training\\checkpoints\\ouro_retention_m2_h2v2_smoke\\progress.json 2>nul']:
    print('--- CMD:', cmd)
    stdin,stdout,stderr=c.exec_command(cmd, timeout=30)
    print(stdout.read().decode(errors='replace'))
    e=stderr.read().decode(errors='replace')
    if e.strip(): print('STDERR:', e)
c.close()
