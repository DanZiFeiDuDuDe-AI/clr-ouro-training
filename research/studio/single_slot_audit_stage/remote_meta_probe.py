import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("100.104.57.18", username="ai training", password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=30)
s = c.open_sftp()
for name in ("train_meta.jsonl", "testA_meta.jsonl", "testB_meta.jsonl"):
    with s.open(r"D:\training\checkpoints\clr_v24_o21a_mr\cache" + "\\" + name, "r") as f:
        print(name)
        for _ in range(2):
            print(f.readline().strip())
s.close()
c.close()
