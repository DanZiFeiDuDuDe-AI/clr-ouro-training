import json
import paramiko

c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("100.104.57.18", username="ai training", password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=30)
s = c.open_sftp()
src = r"D:\training\checkpoints\clr_v24_o3b_e_relation_key\factorized_results.json"
dst = r"D:\training\checkpoints\clr_v24_o3b_e_relation_key\equality_results_compact.json"
with s.open(src, "r") as f:
    data = json.load(f)
compact = {"variants": data.get("variants", {}), "best_variant": data.get("best_variant"), "best_scheme": data.get("best_scheme")}
with s.open(dst, "w") as f:
    f.write(json.dumps(compact, ensure_ascii=False, indent=2))
print("WROTE", dst)
s.close(); c.close()
