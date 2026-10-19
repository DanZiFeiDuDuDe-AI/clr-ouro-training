import paramiko
import sys

host = "100.104.57.18"
user = "ai training"
password = __import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", "")
remote_out = r"D:\training\checkpoints\clr_v24_o3b_g_binding_key"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(host, username=user, password=password, timeout=30)
cmd = (
    'powershell -NoProfile -Command "'
    f"$p='{remote_out}'; if (Test-Path $p) {{ Get-ChildItem -LiteralPath $p -Recurse -File | "
    "Select-Object FullName,Length,LastWriteTime | Sort-Object FullName | Format-Table -AutoSize } else { Write-Output 'MISSING' }"
)
_, stdout, stderr = c.exec_command(cmd)
sys.stdout.buffer.write(stdout.read())
err = stderr.read()
if err:
    sys.stderr.buffer.write(err)
c.close()
