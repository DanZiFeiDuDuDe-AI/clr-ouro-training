import paramiko

local = r"H:\项目\造物工作室\single_slot_audit_stage\a1_v24_o3b_i.py"
remote = r"D:\training\a1_v24_o3b_i.py"
cmd = r'''cd /d D:\training && set "V24_O3BI_OUT=D:\training\checkpoints\clr_v24_o3b_i_mention_write_state" && set "V24_O3BI_DEVICE=cuda" && set "V24_O3BI_PROBE_STEPS=240" && set "V24_O3BI_PAIR_STEPS=280" && set "V24_O3BI_PAIR_N=3500" && py -3 a1_v24_o3b_i.py'''
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("100.104.57.18", username="ai training", password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=30)
s=c.open_sftp(); s.put(local, remote); s.close()
_,o,e=c.exec_command(cmd); print(o.read().decode("utf-8","replace"),end=""); print(e.read().decode("utf-8","replace"),end=""); rc=o.channel.recv_exit_status(); c.close(); raise SystemExit(rc)
