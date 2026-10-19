import paramiko, sys

c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("100.104.57.18", username="ai training", password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=30)
cmd = r'''powershell -NoProfile -Command "$p='D:\training\checkpoints\clr_v24_o3b_g_binding_key'; Get-Date; if(Test-Path $p){ Get-ChildItem -LiteralPath (Join-Path $p 'checkpoints') -File -ErrorAction SilentlyContinue | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize; Get-Process -Id 45916 -ErrorAction SilentlyContinue | Select-Object Id,CPU,PM,StartTime | Format-Table -AutoSize }"'''
_, o, e = c.exec_command(cmd)
sys.stdout.buffer.write(o.read()); sys.stderr.buffer.write(e.read()); c.close()
