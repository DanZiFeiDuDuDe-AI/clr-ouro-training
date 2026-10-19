import paramiko
import sys

local_script = r"H:\项目\造物工作室\single_slot_audit_stage\a1_v24_o3b_h.py"
remote_script = r"D:\training\a1_v24_o3b_h.py"
smoke = len(sys.argv) > 1 and sys.argv[1].lower() == "smoke"
out = r"D:\training\checkpoints\clr_v24_o3b_h_entity_carrier_diag_smoke" if smoke else r"D:\training\checkpoints\clr_v24_o3b_h_entity_carrier_diag"
envcmd = 'set "V24_O3BH_SMOKE=1" && set "V24_O3BH_SKIP_OURO=1"' if smoke else 'set "V24_O3BH_SMOKE=0"'
cmd = f'cd /d D:\\training && set "V24_O3BH_OUT={out}" && set "V24_O3BH_DEVICE=cuda" && set "V24_O3BH_PROBE_STEPS={30 if smoke else 320}" && set "V24_O3BH_PAIR_STEPS={30 if smoke else 350}" && set "V24_O3BH_PAIR_N={300 if smoke else 3500}" && {envcmd} && py -3 a1_v24_o3b_h.py'
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("100.104.57.18", username="ai training", password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=30)
s = c.open_sftp(); s.put(local_script, remote_script); s.close()
stdin, stdout, stderr = c.exec_command(cmd)
print(stdout.read().decode("utf-8", "replace"), end="")
print(stderr.read().decode("utf-8", "replace"), end="")
rc = stdout.channel.recv_exit_status(); c.close(); raise SystemExit(rc)
