from pathlib import Path
import paramiko

local = Path(r"H:\项目\造物工作室\single_slot_audit_stage\o3bf_final")
local.mkdir(parents=True, exist_ok=True)
remote_dir = r"D:\training\checkpoints\clr_v24_o3b_f_binding_diag"
names = [
    "config.json", "git_info.json", "pair_dataset_stats.json", "f0_baseline_reproduction.json",
    "joint_identity_probes.json", "conditional_relation.json", "frozen_k_metrics.json",
    "explicit_binding_results.json", "retrieval_error_decomposition.json", "margin_analysis.json",
    "binding_geometry.json", "collision_matrices.json", "ke_role_analysis.json",
    "gate_evaluation.json", "kk_metric_addendum.json", "training.log", "v24_o3b_f_binding_diag_report.md", "best_matcher.pt",
]
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("100.104.57.18", username="ai training", password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=30)
s = c.open_sftp()
for name in names:
    s.get(remote_dir + "\\" + name, str(local / name))
    print(name)
s.close(); c.close()
