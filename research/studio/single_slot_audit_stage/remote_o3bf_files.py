import paramiko

c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("100.104.57.18", username="ai training", password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=30)
cmd = r'''powershell -NoProfile -Command "Get-ChildItem -LiteralPath 'D:\training\checkpoints\clr_v24_o3b_f_binding_diag' -File -ErrorAction SilentlyContinue | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize"'''
_, out, err = c.exec_command(cmd)
print(out.read().decode("utf-8", "replace"), end="")
print(err.read().decode("utf-8", "replace"), end="")
c.close()
