import paramiko, time
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('cheems', username='ai training', password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=15, banner_timeout=15, auth_timeout=15)
cmd = r'''cd /d D:\training && powershell -Command "$env:H2_OUT_DIR='D:\training\checkpoints\ouro_retention_m2_h2v2_smoke'; $env:H2_STEPS='3'; $env:H2_N_EVAL='20'; $env:H2_MONITOR_EVERY='3'; $env:H2_LOGIT_N='10'; python D:\training\p0h2_v2_train.py"'''
print('running smoke...')
stdin,stdout,stderr=c.exec_command(cmd, timeout=1800)
out_buf=[]; err_buf=[]
while True:
    if stdout.channel.recv_ready():
        b=stdout.channel.recv(65535)
        if b:
            s=b.decode('utf-8', errors='replace')
            print(s, end='')
            out_buf.append(s)
    if stderr.channel.recv_stderr_ready():
        b=stderr.channel.recv_stderr(65535)
        if b:
            s=b.decode('utf-8', errors='replace')
            print('ERR:'+s, end='')
            err_buf.append(s)
    if stdout.channel.exit_status_ready() and not (stdout.channel.recv_ready() or stderr.channel.recv_stderr_ready()):
        break
    time.sleep(0.5)
exit=stdout.channel.recv_exit_status()
print('\nEXIT', exit)
c.close()
