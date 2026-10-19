from pathlib import Path
import paramiko

local = Path(r"H:\项目\造物工作室\single_slot_audit_stage\o3be_final")
local.mkdir(parents=True, exist_ok=True)
remote_dir = r"D:\training\checkpoints\clr_v24_o3b_e_relation_key"
names = [
    "best_model.pt", "final_model.pt", "config.json", "git_info.json",
    "train_log.json", "training.log", "v24_o3b_e_report.md",
    "baseline_reproduction.json", "cosine_baseline.json", "distance_baseline.json",
    "pair_dataset_stats.json", "factorized_results.json", "equality_results_compact.json",
    "geometry_diagnostics.json", "p1_p2_histograms.json", "probe_results.json",
    "retrieval_results.json", "nearest_neighbor_audit.json", "threshold_sensitivity.json",
    "judge_ladder.json", "confusion_matrices.json", "reconnect_readiness.json",
    "factorized_original_best.pt", "factorized_p2_25_best.pt", "factorized_p2_50_best.pt",
]
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("100.104.57.18", username="ai training", password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=30)
s = c.open_sftp()
for name in names:
    s.get(remote_dir + "\\" + name, str(local / name))
    print(name)
s.close(); c.close()
