import paramiko
import sys

mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
out = r"D:\training\checkpoints\clr_v24_o3b_diag_smoke" if mode == "smoke" else r"D:\training\checkpoints\clr_v24_o3b_diag"
env = "V24_O3BD_SMOKE=1" if mode == "smoke" else "V24_O3BD_SMOKE=0"
cmd = f'cd /d D:\\training && set "{env}" && set "V24_O3BD_OUT={out}" && py -3 a1_v24_o3b_diag.py'
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("100.104.57.18", username="ai training", password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=30)
stdin, stdout, stderr = c.exec_command(cmd)
print(stdout.read().decode("utf-8", "replace"), end="")
print(stderr.read().decode("utf-8", "replace"), end="")
rc = stdout.channel.recv_exit_status()
c.close()
raise SystemExit(rc)
