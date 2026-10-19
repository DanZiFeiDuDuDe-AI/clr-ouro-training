import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('cheems', username='ai training', password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=15, banner_timeout=15, auth_timeout=15)
for cmd in ['cd /d D:\\training && dir', 'cd /d D:\\training && dir /b checkpoints']:
    print('--- CMD:', cmd)
    stdin,stdout,stderr=c.exec_command(cmd)
    print(stdout.read().decode(errors='replace'))
    e=stderr.read().decode(errors='replace')
    if e.strip(): print('STDERR:', e)
c.close()
