from pathlib import Path
import paramiko

remote_root = r"D:\training\checkpoints\clr_v24_o3b_i_mention_write_state"
local_root = Path(r"H:\项目\造物工作室\single_slot_audit_stage\o3bi_final")
local_root.mkdir(parents=True, exist_ok=True)
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy()); c.connect("100.104.57.18",username="ai training",password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""),timeout=30)
s=c.open_sftp()
def walk(rdir, ldir):
    ldir.mkdir(parents=True, exist_ok=True)
    for ent in s.listdir_attr(rdir):
        rp=rdir+"\\"+ent.filename; lp=ldir/ent.filename
        if ent.st_mode & 0o040000: walk(rp,lp)
        else: s.get(rp,str(lp))
walk(remote_root,local_root); s.close(); c.close(); print(local_root)
