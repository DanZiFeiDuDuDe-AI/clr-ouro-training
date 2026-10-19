import paramiko
host='cheems'; port=22
for user in ['ai training','ai','aitraining','training']:
    try:
        c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(host, port=port, username=user, password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=8, banner_timeout=8, auth_timeout=8)
        stdin,stdout,stderr=c.exec_command('whoami')
        out=stdout.read().decode(errors='replace').strip()
        err=stderr.read().decode(errors='replace').strip()
        print('OK user=',repr(user),'whoami=',out,err)
        c.close()
        break
    except Exception as e:
        print('FAIL',repr(user),type(e).__name__,str(e)[:160])
