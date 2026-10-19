import paramiko
import sys

mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
local_script = r"H:\项目\造物工作室\single_slot_audit_stage\a1_v24_o3b_e.py"
remote_script = r"D:\training\a1_v24_o3b_e.py"
if mode == "smoke":
    out = r"D:\training\checkpoints\clr_v24_o3b_e_relation_key_smoke"
    env = "V24_O3BE_SMOKE=1"
else:
    out = r"D:\training\checkpoints\clr_v24_o3b_e_relation_key"
    env = "V24_O3BE_SMOKE=0"
cmd = f'cd /d D:\\training && set "{env}" && set "V24_O3BE_OUT={out}" && py -3 a1_v24_o3b_e.py'
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("100.104.57.18", username="ai training", password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=30)
s = c.open_sftp(); s.put(local_script, remote_script); s.close()
stdin, stdout, stderr = c.exec_command(cmd)
print(stdout.read().decode("utf-8", "replace"), end="")
print(stderr.read().decode("utf-8", "replace"), end="")
rc = stdout.channel.recv_exit_status()
c.close()
raise SystemExit(rc)
