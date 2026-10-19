from pathlib import Path
import paramiko
remote_root=r"D:\training\checkpoints\clr_v24_o3b_i_mention_write_state"; local_root=Path(r"H:\项目\造物工作室\single_slot_audit_stage\o3bi_final"); local_root.mkdir(parents=True,exist_ok=True)
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy()); c.connect("100.104.57.18",username="ai training",password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""),timeout=30); s=c.open_sftp()
for ent in s.listdir_attr(remote_root):
    if ent.filename.endswith('.npz'): continue
    s.get(remote_root+'\\'+ent.filename,str(local_root/ent.filename))
s.close(); c.close(); print('downloaded JSON/report artifacts')
