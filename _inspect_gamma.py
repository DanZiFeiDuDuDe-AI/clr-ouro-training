"""Fetch M1/M2-v1/M2-v2 checkpoints from cheems and compare gamma movement."""
import sys, os, warnings
import paramiko, torch

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("cheems", username="ai training", password="REPLACE_ME", timeout=30, banner_timeout=30, auth_timeout=30)

paths = {
    "M1": "D:/training/checkpoints/ouro_retention_m1/best_model.pt",
    "M2v1": "D:/training/checkpoints/ouro_retention_m2_d1/best_model.pt",
    "M2v2": "D:/training/checkpoints/ouro_retention_m2_d2/best_model.pt",
}
local_dir = "D:/LoopLM/tmp/ckpt_inspect"
os.makedirs(local_dir, exist_ok=True)

sftp = c.open_sftp()
local = {}
for tag, rp in paths.items():
    lp = os.path.join(local_dir, tag + ".pt")
    sftp.get(rp, lp)
    local[tag] = lp
    print("fetched %s" % lp)
sftp.close()
c.close()

m1 = torch.load(local["M1"], map_location="cpu", weights_only=False)
m2v1 = torch.load(local["M2v1"], map_location="cpu", weights_only=False)
m2v2 = torch.load(local["M2v2"], map_location="cpu", weights_only=False)

for tag, ck in [("M1", m1), ("M2v1", m2v1), ("M2v2", m2v2)]:
    g = ck["retention_gamma"]
    print(tag, "gamma range", min(float(v.min()) for v in g.values()), max(float(v.max()) for v in g.values()))
    if tag != "M1":
        for k in g.keys():
            d = (g[k] - m1["retention_gamma"][k]).abs().max().item()
            if d > 1e-9:
                print("  layer", k, "gamma_logit max diff vs M1:", d)
print("M2v1 vs M1 total gamma max diff:", max((m2v1["retention_gamma"][k]-m1["retention_gamma"][k]).abs().max().item() for k in m1["retention_gamma"]))
print("M2v2 vs M1 total gamma max diff:", max((m2v2["retention_gamma"][k]-m1["retention_gamma"][k]).abs().max().item() for k in m1["retention_gamma"]))
