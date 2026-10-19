import paramiko
import sys

local_script = r"H:\项目\造物工作室\single_slot_audit_stage\a1_v24_o3b_f_kk_addendum.py"
remote_script = r"D:\training\a1_v24_o3b_f_kk_addendum.py"
out = r"D:\training\checkpoints\clr_v24_o3b_f_binding_diag"
smoke = len(sys.argv) > 1 and sys.argv[1].lower() == "smoke"
env = "V24_O3BF_SMOKE=1" if smoke else "V24_O3BF_SMOKE=0"
steps = "V24_O3BF_MATCHER_STEPS=20" if smoke else "V24_O3BF_MATCHER_STEPS=800"
cmd = f'cd /d D:\\training && set "{env}" && set "{steps}" && set "V24_O3BF_OUT={out}" && py -3 a1_v24_o3b_f_kk_addendum.py'
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("100.104.57.18", username="ai training", password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=30)
s = c.open_sftp(); s.put(local_script, remote_script); s.close()
stdin, stdout, stderr = c.exec_command(cmd)
print(stdout.read().decode("utf-8", "replace"), end="")
print(stderr.read().decode("utf-8", "replace"), end="")
rc = stdout.channel.recv_exit_status(); c.close(); raise SystemExit(rc)
