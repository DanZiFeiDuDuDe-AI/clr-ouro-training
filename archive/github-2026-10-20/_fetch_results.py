"""Fetch matched eval results and gamma diagnostics from cheems."""
import os, sys, warnings, json
import paramiko, torch

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("cheems", username="ai training", password="REPLACE_ME", timeout=30, banner_timeout=30, auth_timeout=30)

sftp = c.open_sftp()
remote_files = {
    "matched_eval_results.json": "D:/training/matched_eval_results.json",
    "p0h_post_distill_eval_gammafix_gamma.json": "D:/training/p0h_post_distill_eval_gammafix_gamma.json",
    "p0h_post_distill_eval_gammafix_norm.json": "D:/training/p0h_post_distill_eval_gammafix_norm.json",
    "gammafix_gamma_best.pt": "D:/training/checkpoints/ouro_retention_m2_gammafix/best_model.pt",
    "gammafix_norm_best.pt": "D:/training/checkpoints/ouro_retention_m2_gammafix_norm/best_model.pt",
    "gammafix_norm_diag.json": "D:/training/checkpoints/ouro_retention_m2_gammafix_norm/gamma_diagnostics.json",
}
local_dir = "D:/LoopLM/tmp/ckpt_inspect2"
os.makedirs(local_dir, exist_ok=True)
for name, rp in remote_files.items():
    lp = os.path.join(local_dir, name)
    try:
        sftp.get(rp, lp)
        print("fetched", name)
    except Exception as e:
        print("FAILED", name, str(e))
sftp.close()
c.close()

print("\n=== matched_eval_results.json ===")
with open(os.path.join(local_dir, "matched_eval_results.json"), encoding="utf-8") as f:
    matched = json.load(f)
print(json.dumps(matched, indent=2))

print("\n=== gammafix gamma checkpoint vs init ===")
import torch
ck = torch.load(os.path.join(local_dir, "gammafix_gamma_best.pt"), map_location="cpu", weights_only=False)
for k, g in ck["retention_gamma"].items():
    print(k, g.tolist())
print("gamma_param_mode:", ck.get("gamma_param_mode"))
print("final_loss:", ck.get("final_loss"))

print("\n=== gammafix norm diagnostics (first 2000 chars) ===")
with open(os.path.join(local_dir, "gammafix_norm_diag.json"), encoding="utf-8") as f:
    diag = json.load(f)
print(json.dumps(diag, indent=2)[:2500])
