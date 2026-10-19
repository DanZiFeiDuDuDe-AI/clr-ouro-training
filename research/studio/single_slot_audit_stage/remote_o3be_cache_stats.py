import paramiko

c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("100.104.57.18", username="ai training", password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=30)
cmd = r'''powershell -NoProfile -Command "$p='D:\training\checkpoints\clr_v24_o21a_mr\cache'; $fs=Get-ChildItem -LiteralPath $p -File; [pscustomobject]@{files=$fs.Count; bytes=[int64](($fs|Measure-Object Length -Sum).Sum); names=($fs.Name -join ',')} | ConvertTo-Json -Compress"'''
_, out, err = c.exec_command(cmd)
print(out.read().decode("utf-8", "replace"), end="")
print(err.read().decode("utf-8", "replace"), end="")
c.close()
