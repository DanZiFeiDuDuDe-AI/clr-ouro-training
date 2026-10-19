"""Check running python process details + distill outputs on cheems."""
import sys, warnings
import paramiko

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("cheems", username="ai training", password="REPLACE_ME", timeout=20, banner_timeout=30, auth_timeout=30)

cmds = [
    'tasklist /FI "IMAGENAME eq python.exe" /V /FO LIST',
    'wmic process where "name=\'python.exe\'" get ProcessId,CreationDate,CommandLine /FORMAT:LIST',
    'if exist D:\\training\\distill.log (type D:\\training\\distill.log) else (echo NO_DISTILL_LOG)',
    'if exist D:\\training\\distill_log.json (type D:\\training\\distill_log.json) else (echo NO_DISTILL_LOG_JSON)',
    'dir D:\\training\\checkpoints\\ouro_retention_m2_d1',
    'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader',
]
try:
    for cmd in cmds:
        print("=== $ " + cmd)
        stdin, stdout, stderr = c.exec_command(cmd, timeout=60)
        out = stdout.read().decode("utf-8", "replace").strip()
        err = stderr.read().decode("utf-8", "replace").strip()
        rc = stdout.channel.recv_exit_status()
        print(out)
        if err:
            print("[stderr]", err)
        print("[rc=%d]" % rc)
finally:
    c.close()
