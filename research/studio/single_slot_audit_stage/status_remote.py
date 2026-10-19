import paramiko
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("100.104.57.18", username="ai training", password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=30)
_, o, e = c.exec_command('powershell -NoProfile -Command "Get-Process python | Select-Object Id,CPU,StartTime,PM | Format-Table -AutoSize"')
data = o.read().decode("utf-8", "replace")
import sys
sys.stdout.buffer.write(data.encode("ascii", "replace"))
c.close()
