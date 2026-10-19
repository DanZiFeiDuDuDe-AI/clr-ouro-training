from pathlib import Path
import paramiko

local_dir = Path(r"H:\项目\造物工作室\single_slot_audit_stage\o3bf_final")
remote_dir = r"D:\training\checkpoints\clr_v24_o3b_f_binding_diag"
names = ["v24_o3b_f_binding_diag_report.md", "gate_evaluation.json", "f3_protocol_audit.json", "config.json", "..\\a1_v24_o3b_f.py"]
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("100.104.57.18", username="ai training", password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=30)
s = c.open_sftp()
for name in names:
    local_path = (local_dir / name).resolve()
    remote_path = remote_dir + "\\" + name
    if name.startswith(".."):
        remote_path = r"D:\training\a1_v24_o3b_f.py"
    s.put(str(local_path), remote_path)
    print(name)
s.close(); c.close()
