import paramiko

local = r"H:\项目\造物工作室\single_slot_audit_stage\o3bh_final\v24_o3b_h_entity_carrier_diag_report.md"
remote = r"D:\training\checkpoints\clr_v24_o3b_h_entity_carrier_diag\v24_o3b_h_entity_carrier_diag_report.md"
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("100.104.57.18", username="ai training", password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=30)
s = c.open_sftp(); s.put(local, remote); s.close(); c.close(); print(remote)
