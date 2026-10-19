"""Check cheems training state before restarting 0H distillation."""
import sys, warnings
import paramiko

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("cheems", username="ai training", password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=20, banner_timeout=30, auth_timeout=30)

cmds = [
    'hostname',
    'whoami',
    'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader',
    'tasklist /FI "IMAGENAME eq python.exe" /FO CSV',
    'tasklist /FI "IMAGENAME eq python3.exe" /FO CSV',
    'if exist D:\\training\\distill.log (type D:\\training\\distill.log) else (echo NO_DISTILL_LOG)',
    'dir D:\\training\\checkpoints /B',
    'if exist D:\\training\\checkpoints\\ouro_retention_m2_d1\\best_model.pt (echo M2_EXISTS) else (echo NO_M2)',
    'dir D:\\training\\teacher_cache_manifest.json',
    'dir D:\\training\\weight_migration_manifest.json',
    'dir D:\\training\\p0g_pre_distill_eval.json',
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
