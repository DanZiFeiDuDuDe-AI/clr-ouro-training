import os
import posixpath
import paramiko

remote = r"D:\\training\\checkpoints\\clr_v24_o3b_diag"
local = r"H:\\项目\\造物工作室\\single_slot_audit_stage\\o3bd_final"

c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("100.104.57.18", username="ai training", password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=30)
s = c.open_sftp()
os.makedirs(local, exist_ok=True)
for attr in s.listdir_attr(remote):
    name = attr.filename
    if name.endswith((".json", ".md", ".log", ".pt")):
        s.get(posixpath.join(remote, name), os.path.join(local, name))
s.close(); c.close()
print("downloaded", local)
