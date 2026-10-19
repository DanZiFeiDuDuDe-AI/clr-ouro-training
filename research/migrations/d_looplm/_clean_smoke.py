import paramiko, time
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('cheems', username='ai training', password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=15, banner_timeout=15, auth_timeout=15)
cmd = r'cd /d D:\training && if exist "D:\training\checkpoints\ouro_retention_m2_h2v2_smoke" rmdir /s /q "D:\training\checkpoints\ouro_retention_m2_h2v2_smoke" && dir /b checkpoints'
stdin,stdout,stderr=c.exec_command(cmd)
print(stdout.read().decode(errors='replace'))
print(stderr.read().decode(errors='replace'))
c.close()
