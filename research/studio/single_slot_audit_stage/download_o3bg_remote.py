import os
from pathlib import Path

import paramiko


HOST = "100.104.57.18"
USER = "ai training"
PASSWORD = __import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", "")
REMOTE_ROOT = r"D:\training\checkpoints\clr_v24_o3b_g_binding_key"
LOCAL_ROOT = Path(r"H:\项目\造物工作室\single_slot_audit_stage\o3bg_final")


def download_tree(sftp, remote_dir, local_dir):
    local_dir.mkdir(parents=True, exist_ok=True)
    for item in sftp.listdir_attr(remote_dir):
        rpath = remote_dir.rstrip("\\") + "\\" + item.filename
        lpath = local_dir / item.filename
        mode = item.st_mode
        if mode & 0o40000:
            download_tree(sftp, rpath, lpath)
        else:
            sftp.get(rpath, str(lpath))
            print(lpath)


client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASSWORD, timeout=30)
sftp = client.open_sftp()
download_tree(sftp, REMOTE_ROOT, LOCAL_ROOT)
sftp.close()
client.close()
