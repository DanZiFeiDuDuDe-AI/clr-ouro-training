import paramiko

local = r"H:\项目\造物工作室\single_slot_audit_stage\a1_v24_o3b_j.py"
remote = r"D:\training\a1_v24_o3b_j.py"
cmd = r'''cd /d D:\training && set "V24_O3BJ_OUT=D:\training\checkpoints\clr_v24_o3b_j_projection_localization" && set "V24_O3BJ_DEVICE=cuda" && set "V24_O3BJ_PROBE_STEPS=180" && set "V24_O3BJ_MATCH_STEPS=220" && set "V24_O3BJ_PAIR_N=3000" && set "V24_O3BJ_TRAIN_PAIR_N=6000" && py -3 a1_v24_o3b_j.py'''
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("100.104.57.18", username="ai training", password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=30)
s = c.open_sftp(); s.put(local, remote); s.close()
stdin, stdout, stderr = c.exec_command(cmd)
print(stdout.read().decode("utf-8", "replace"), end="")
print(stderr.read().decode("utf-8", "replace"), end="")
rc = stdout.channel.recv_exit_status(); c.close(); raise SystemExit(rc)
