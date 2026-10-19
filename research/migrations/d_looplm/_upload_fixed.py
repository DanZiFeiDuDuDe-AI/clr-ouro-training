import paramiko
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('cheems', username='ai training', password=__import__("os").environ.get("LOOPLM_LEGACY_SSH_PASSWORD", ""), timeout=15, banner_timeout=15, auth_timeout=15)
sftp=c.open_sftp()
# backup previous train script if not already
import os
local_fixed = r'D:\LoopLM\remote_mirror\p0h2_v2_train_fixed.py'
# upload to remote
sftp.put(local_fixed, 'D:/training/p0h2_v2_train.py')
print('uploaded fixed p0h2_v2_train.py')
# verify size
st=sftp.stat('D:/training/p0h2_v2_train.py')
print('remote size', st.st_size)
sftp.close(); c.close()
