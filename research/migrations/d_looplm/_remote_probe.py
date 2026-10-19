import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('cheems', username='ai training', password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=15, banner_timeout=15, auth_timeout=15)
for cmd in ['tasklist /FI "IMAGENAME eq python.exe" /V', 'type D:\\training\\h2_pid.txt', 'dir /b D:\\training\\checkpoints\\ouro_retention_m2_h2v2', 'dir /b D:\\training\\checkpoints\\ouro_retention_m2_d3', 'type D:\\training\\teacher_cache_v2_manifest.json']:
    print('--- CMD:', cmd)
    stdin,stdout,stderr=c.exec_command(cmd)
    print(stdout.read().decode(errors='replace'))
    e=stderr.read().decode(errors='replace')
    if e.strip(): print('STDERR:', e)
c.close()
