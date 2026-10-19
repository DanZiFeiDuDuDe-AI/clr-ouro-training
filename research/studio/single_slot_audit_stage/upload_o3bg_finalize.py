from pathlib import Path
import paramiko

HOST = "100.104.57.18"
USER = "ai training"
PASSWORD = __import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", "")
REMOTE_ROOT = r"D:\training\checkpoints\clr_v24_o3b_g_binding_key"
LOCAL_ROOT = Path(r"H:\项目\造物工作室\single_slot_audit_stage\o3bg_final")

client = paramiko.SSHClient(); client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASSWORD, timeout=30)
sftp = client.open_sftp()
try:
    sftp.mkdir(REMOTE_ROOT + "\\plots")
except IOError:
    pass
for rel in ["v24_o3b_g_binding_key_report.md", "plots/variant_f1_p2.svg", "plots/p1_p2_overlap_testB.svg", "plots/collision_matrix_best_testB.svg"]:
    local = LOCAL_ROOT / rel
    remote = REMOTE_ROOT + "\\" + rel.replace("/", "\\")
    sftp.put(str(local), remote)
    print(remote)
sftp.close(); client.close()
