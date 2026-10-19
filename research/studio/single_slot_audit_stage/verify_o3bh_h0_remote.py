import json
from pathlib import Path
import paramiko

local_script = r"H:\项目\造物工作室\single_slot_audit_stage\verify_o3bh_h0.py"
remote_script = r"D:\training\verify_o3bh_h0.py"
cmd = r'''cd /d D:\training && py -3 verify_o3bh_h0.py'''
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("100.104.57.18", username="ai training", password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=30)
s=c.open_sftp(); s.put(local_script, remote_script); s.close()
_,o,e=c.exec_command(cmd); print(o.read().decode("utf-8","replace"),end=""); print(e.read().decode("utf-8","replace"),end=""); rc=o.channel.recv_exit_status(); c.close(); raise SystemExit(rc)
