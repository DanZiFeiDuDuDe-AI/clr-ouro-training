import paramiko
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("100.104.57.18", username="ai training", password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=30)
cmd = r'''powershell -NoProfile -Command "Get-ChildItem -Force 'D:\training\checkpoints\clr_v24_o3b_h_entity_carrier_diag' -Recurse -File -ErrorAction SilentlyContinue | Select-Object FullName,Length,LastWriteTime | Format-Table -AutoSize"'''
_, o, e = c.exec_command(cmd)
print(o.read().decode("utf-8", "replace").encode("ascii", "replace").decode("ascii"), end="")
print(e.read().decode("utf-8", "replace").encode("ascii", "replace").decode("ascii"), end="")
c.close()
