import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('cheems', username='ai training', password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=15, banner_timeout=15, auth_timeout=15)
stdin,stdout,stderr=c.exec_command('taskkill /PID 32592 /F')
print(stdout.read().decode(errors='replace'))
print(stderr.read().decode(errors='replace'))
c.close()
