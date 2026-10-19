import paramiko

local_script = r"H:\项目\造物工作室\single_slot_audit_stage\a1_v24_o3b_g_post.py"
remote_script = r"D:\training\a1_v24_o3b_g_post.py"
cmd = r'cd /d D:\training && py -3 a1_v24_o3b_g_post.py'
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("100.104.57.18", username="ai training", password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=30)
s = c.open_sftp(); s.put(local_script, remote_script); s.close()
stdin, stdout, stderr = c.exec_command(cmd)
print(stdout.read().decode("utf-8", "replace"), end="")
print(stderr.read().decode("utf-8", "replace"), end="")
rc = stdout.channel.recv_exit_status(); c.close(); raise SystemExit(rc)
